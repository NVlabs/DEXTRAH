import os
import pathlib
import yaml
import time

import torch
from torch.cuda.amp import autocast
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.algos_torch import torch_ext

from dextrah_lab.distillation.a2c_with_aux_transformer_stereo import A2CBuilder as A2CWithAuxTransformerCNNStereoBuilder

def load_param_dict(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def adjust_state_dict_keys(checkpoint_state_dict, model_state_dict):
    adjusted_state_dict = {}
    num_elems = 0
    for key, value in checkpoint_state_dict.items():
        num_elems += value.numel()
        # If the key is in the model's state_dict, use it directly
        if key in model_state_dict:
            adjusted_state_dict[key] = value
        else:
            # Try inserting '_orig_mod' in different positions based on key structure
            parts = key.split(".")
            new_key_with_orig_mod = None
            
            # Try inserting '_orig_mod' before the last layer index for different cases
            parts.insert(2, "_orig_mod")
            new_key_with_orig_mod = ".".join(parts)
            
            # If adding '_orig_mod' matches a key in the model, use the modified key
            if new_key_with_orig_mod in model_state_dict:
                adjusted_state_dict[new_key_with_orig_mod] = value
            else:
                # check if removing orig_mod works
                key_no_orig_mod = key.replace("_orig_mod.", "")
                if key_no_orig_mod in model_state_dict:
                    adjusted_state_dict[key_no_orig_mod] = value
                else:
                    # Log the key that couldn't be matched, for debugging purposes
                    print(f"Could not match key: {key} -> {new_key_with_orig_mod}")
                    # If neither works, retain the original key as a fallback
                    adjusted_state_dict[key] = value
        
    print(f"Number of elements in adjusted state dict: {num_elems}")
    return adjusted_state_dict

class RLGamesPolicy:
    def __init__(
        self, cfg_path, img_shape, num_proprio_obs,
        num_actions, ckpt_path=None, device="cuda"
    ):
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.device = device

        # read the yaml file
        network_params = load_param_dict(cfg_path)["params"]
        self.num_proprio_obs = num_proprio_obs
        self.img_shape = img_shape

        # build the model config
        normalize_value = network_params["config"]["normalize_value"]
        normalize_input = network_params["config"]["normalize_input"]
        model_config = {
            "actions_num": num_actions,
            "input_shape": (num_proprio_obs,),
            "num_seqs": 2,
            "value_size": 1,
            'normalize_value': normalize_value,
            'normalize_input': normalize_input,
            'num_envs': 2,
        }

        # build the model
        builder = ModelBuilder()
        network = builder.load(network_params)
        self.model = network.build(model_config).to(self.device)
        self.model.eval()

        # load checkpoint if available
        if ckpt_path is not None:
            weights = torch_ext.load_checkpoint(ckpt_path)
            weights["model"] = adjust_state_dict_keys(
                weights["model"],
                self.model.state_dict()
            )
            self.model.load_state_dict(weights["model"])
            if normalize_input and 'running_mean_std' in weights:
                self.model.running_mean_std.load_state_dict(
                    weights["running_mean_std"]
                )

        # dummy variable, this doesn't actually contain prev actions
        # need this bc of rl_games weirdness...
        self.dummy_prev_actions = torch.zeros(
            (2, num_actions), dtype=torch.float32
        ).to(self.device)



    def step(self, proprio, left_img, right_img):
        # package observations
        batch_dict = {
            "is_train": True,
            "obs": proprio.repeat(2, 1),
            "img_left": left_img.repeat(2, 1, 1, 1),
            "img_right": right_img.repeat(2, 1, 1, 1),
            "prev_actions": self.dummy_prev_actions
        }
        # add extra information for RNNs
        if self.model.is_rnn():
            batch_dict["seq_length"] = 1
            batch_dict["rnn_masks"] = None

        # step through model
        res_dict = self.model(batch_dict)
        mus = res_dict["mus"][0:1]
        sigmas = res_dict["sigmas"][0:1]

        position = res_dict["rnn_states"][1]['object_pos'][0:1]

        distr = torch.distributions.Normal(mus, sigmas, validate_args=False)
        selected_action = mus

        return {
            "mus": mus,
            "sigmas": sigmas,
            "obj_pos": position,
            "selected_action": selected_action
        }


    def setup_cuda_graph(self):
        dummy_proprio = torch.randn(1, self.num_proprio_obs).to(self.device)
        dummy_left_img = torch.randn(1, *self.img_shape).to(self.device)
        dummy_right_img = torch.randn(1, *self.img_shape).to(self.device)

        for _ in range(3):
            with torch.no_grad():
                self.step(
                    dummy_proprio, dummy_left_img, dummy_right_img
                )

        self.reset()
        self.cuda_graph = torch.cuda.CUDAGraph()

        self.static_proprio = torch.empty_like(dummy_proprio, device=self.device)
        self.static_left_img = torch.empty_like(dummy_left_img, device=self.device)
        self.static_right_img = torch.empty_like(dummy_right_img, device=self.device)

        with torch.cuda.graph(self.cuda_graph):
            with torch.no_grad():
                self._policy_out = self.step(
                    self.static_proprio,
                    self.static_left_img,
                    self.static_right_img,
                )
        
    def step_cuda_graph(self, proprio, left_img, right_img):
        self.static_proprio.copy_(proprio)
        self.static_left_img.copy_(left_img)
        self.static_right_img.copy_(right_img)
        self.cuda_graph.replay()

        policy_out = self._policy_out
        policy_out = {
            "mus": self._policy_out["mus"].clone(),
            "sigmas": self._policy_out["sigmas"].clone(),
            "obj_pos": self._policy_out["obj_pos"].clone(),
        }
        policy_out["selected_action"] = torch.distributions.Normal(
            policy_out["mus"],
            policy_out["sigmas"]
        ).sample()
    
        return policy_out

    def reset(self):
        self.model.a2c_network.reset_idx(torch.arange(2))


def main():
    # get path to config file
    parent_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
    agent_cfg_folder = "dextrah_lab/tasks/dextrah_kuka_allegro/agents"
    student_cfg_path = os.path.join(
        parent_path,
        agent_cfg_folder,
        "rl_games_ppo_transformer_stereo.yaml"
    )

    # get path to checkpoint
    # NOTE: This assumes that in the root directory of dextrah_lab, the checkpoint is stored in a folder called pretrained_ckpts
    student_ckpt = "pretrained_ckpts/dextrah_student_85000_iters.pth"
    student_ckpt_path = os.path.join(
        parent_path,
        student_ckpt
    )
    # student_ckpt_path = "/home/ritviks/workspace/git/dextrah_lab/dextrah_lab/distillation/runs/Dextrah-Kuka-Allegro_16-11-46-11/nn/dextrah_student_10000_iters.pth"
    # student_ckpt_path = "/home/ritviks/workspace/dextrah_distillation_results/dextrah_rgb_asym1_kl/model/nn/dextrah_student_50000_iters_1.pth"
    student_ckpt_path = None
    student_ckpt_path = "/home/ritviks/workspace/git/dextrah_lab/dextrah_lab/distillation/runs/Dextrah-Kuka-Allegro_11-10-42-17/nn/dextrah_student_1000_iters.pth"

    # register our custom model with the rl_games model builder
    model_builder.register_network("a2c_aux_transformer_stereo", A2CWithAuxTransformerCNNStereoBuilder)

    num_proprio_obs = 159
    num_actions = 11
    img_shape = (3, 240, 320)
    # create the model
    policy = RLGamesPolicy(
        cfg_path=student_cfg_path,
        img_shape=img_shape,
        num_proprio_obs=num_proprio_obs,
        num_actions=num_actions,
        ckpt_path=student_ckpt_path,
    )

    dummy_proprio = torch.randn(1, num_proprio_obs).to(policy.device)
    dummy_left_img = torch.randn(1, *img_shape).to(policy.device)
    dummy_right_img = torch.randn(1, *img_shape).to(policy.device)
    policy.reset()

    num_samples = 5
    elapsed_time = 0
    dummy_proprios = [
        torch.randn(1, num_proprio_obs).to(policy.device)
        for _ in range(num_samples)
    ]
    dummy_left_imgs = [
        torch.randn(1, *img_shape).to(policy.device)
        for _ in range(num_samples)
    ]
    dummy_right_imgs = [
        torch.randn(1, *img_shape).to(policy.device)
        for _ in range(num_samples)
    ]
    for i in range(num_samples):
        dummy_proprio = dummy_proprios[i]
        dummy_left_img = dummy_left_imgs[i]
        dummy_right_img = dummy_right_imgs[i]
        # forward
        start = time.time()
        policy_out_normal = policy.step(
            proprio=dummy_proprio,
            left_img=dummy_left_img,
            right_img=dummy_right_img,
        )
        policy.reset()
        torch.cuda.synchronize()
        end = time.time()
        elapsed_time += end - start
        print(policy_out_normal)
    print(f"Average time taken normally: {elapsed_time / num_samples}")
        
    # Capture CUDA graph for model inference
    torch.cuda.synchronize()

    print("CUDA graph loaded")
    # torch.set_float32_matmul_precision('high')

    policy.reset()
    policy.setup_cuda_graph()
    for i in range(num_samples):
        dummy_proprio = dummy_proprios[i]
        dummy_left_img = dummy_left_imgs[i]
        dummy_right_img = dummy_right_imgs[i]
        t1 = time.time()
        policy_out_cuda_graph = policy.step_cuda_graph(
            dummy_proprio, dummy_left_img, dummy_right_img
        )
        torch.cuda.synchronize()
        t2 = time.time()
        print(f"Time taken: {t2 - t1}")
        policy.reset()
        print(policy_out_cuda_graph)



if __name__ == "__main__":
    main()
