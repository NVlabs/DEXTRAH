import torch
from torchvision import transforms
import random
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as vutils
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import pathlib
from PIL import Image
import glob
import time
import math

from rl_games.common import a2c_common 
from rl_games.algos_torch import torch_ext 
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs 
from rl_games.algos_torch import central_value 
from rl_games.common import common_losses 
from rl_games.common import datasets
from rl_games.common import tr_helpers
from rl_games.common import vecenv
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common import schedulers
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.a2c_common import swap_and_flatten01
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.model_builder import ModelBuilder
from datetime import datetime
from tensorboardX import SummaryWriter
import wandb

from depth_augs import DepthAug
from rgb_augs import RgbAug
from typing import Dict

from isaaclab.sensors import save_images_to_file


def l2(model, target):
    """Computes the L2 norm between model and target.
    """

    return torch.norm(model - target, p=2, dim=-1)

def weighted_l2(model, target, weights):
    return torch.sum((model - target) * (weights * (model - target)), dim=-1) ** 0.5


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action

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



class Dagger:
    def __init__(self, env, config, summaries_dir, nn_dir):
        self.world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
        self.rank = int(os.environ['RANK'])  # Global rank of this process
        self.local_rank = int(os.environ['LOCAL_RANK']) # local rank of the process 
        torch.cuda.set_device(self.local_rank)
        wp.set_device(f"cuda:{self.local_rank}")

        self.env = env
        self.ov_env = env.env
        self.num_envs = self.ov_env.num_envs
        self.num_actions = self.ov_env.num_actions
        self.device = self.local_rank
        self.config = config
        self.student_network_params = self.load_param_dict(self.config["student"]["cfg"])["params"]
        self.use_data_aug = self.config["student"]["data_aug"]
        self.teacher_network_params = self.load_param_dict(self.config["teacher"]["cfg"])["params"]
        self.student_network = self.load_networks(self.student_network_params)
        self.teacher_network = self.load_networks(self.teacher_network_params)

        self.value_size = 1
        self.horizon_length = self.student_network_params["config"]["horizon_length"]
        self.normalize_value = self.student_network_params["config"]["normalize_value"]
        self.normalize_input = self.student_network_params["config"]["normalize_input"]

        # get student and teacher models
        self.num_actions_student = self.num_actions
        self.student_model_config = {
            "actions_num": self.num_actions_student,
            "input_shape": (self.ov_env.num_observations,),
            "batch_size": self.num_envs,
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.teacher_model_config = {
            "actions_num": self.num_actions,
            "input_shape": (self.ov_env.num_teacher_observations,),
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            'normalize_value': self.normalize_value, 
            'normalize_input': self.normalize_input,
        }
        self.student_model = self.student_network.build(self.student_model_config).to(self.device)
        for param in self.student_model.parameters():
            dist.broadcast(param.data, src=0)
        self.student_model_ddp = DDP(self.student_model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.teacher_model = self.teacher_network.build(self.teacher_model_config).to(self.device)
        self.warm_up_lr = 1e-5
        self.peak_lr = 1e-3
        params = [{"params": self.student_model_ddp.parameters(), "lr": self.warm_up_lr, "eps": 1e-8}]
        # self.optimizer = torch.optim.Adam(params)
        self.optimizer = torch.optim.Adam(self.student_model_ddp.parameters(), lr=1e-4, eps=1e-8)
        self.warmup_epochs = 2000
        self.max_epochs = 100_000
        self.num_cycles = 1.
        # self.scheduler = self.cosine_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=self.warmup_epochs,
        #     num_training_steps=self.max_epochs,
        #     num_cycles=self.num_cycles
        # )
        self.num_warmup_steps = 1000
        self.num_iters = 100_000

        # load weights for student and teacher
        if self.config["student"]["ckpt"] is not None:
            self.set_weights(self.config["student"]["ckpt"], "student")
        if self.config["teacher"]["ckpt"] is not None:
            self.set_weights(self.config["teacher"]["ckpt"], "teacher")
        # get the observation type of the student and teacher
        self.student_obs_type = self.config["student"]["obs_type"]
        self.teacher_obs_type = self.config["teacher"]["obs_type"]
        self.is_rnn = self.student_model.is_rnn()
        self.is_teacher_rnn = self.teacher_model.is_rnn()
        if self.is_rnn:
            self.seq_length = self.student_network_params["config"]["seq_length"]
            self.seq_length = 1
            print("USING RNN")
        if self.is_teacher_rnn:
            print("USING TEACHER RNN")
        if hasattr(self.student_model.a2c_network, "is_aux") and self.student_model.a2c_network.is_aux:
            self.is_aux = True
            print("USING AUX")
        else:
            self.is_aux = False
        self.step_student_actions = True
        self.play_policy = self.config["play_policy"]
        if self.play_policy is True:
            self.step_student_actions = True

        # logging
        self.games_to_track = 100
        self.frame = 0
        self.epoch_num = 0
        self.game_rewards = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.device)

        if self.rank == 0:
            self.writer = SummaryWriter(summaries_dir)
            self.use_wandb = False
            parent_path = str(pathlib.Path(__file__).parent.resolve())
            summaries_dir = os.path.join(parent_path, summaries_dir)
            self.nn_dir = os.path.join(parent_path, nn_dir)
            if self.use_wandb:
                wandb.login(key=os.environ["WANDB_API_KEY"])
                # wandb.tensorboard.patch(root_logdir=summaries_dir)
                wandb.init(
                    project=os.environ["WANDB_PROJECT"],
                    entity=os.environ["WANDB_ENTITY"],
                    name=os.environ["WANDB_NAME"],
                    notes=os.environ["WANDB_NOTES"],
                    # sync_tensorboard=True,
                )
        else:
            self.use_wandb = False
        self.scaler = GradScaler()
        wp.init()
        self.depth_aug = DepthAug(f"cuda:{self.local_rank}")
        self.use_depth_aug = self.ov_env.cfg.aug_depth
        self.img_aug_type = self.ov_env.cfg.img_aug_type
        self.aux_coeff = self.ov_env.cfg.aux_coeff
        self.depth_aug_cfg = self.ov_env.cfg.depth_randomization_cfg_dict
        self.stereo = self.ov_env.cfg.simulate_stereo
        self.viz_imgs = False
        if self.viz_imgs:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))

            x = np.linspace(0, 50., num=self.ov_env.cfg.img_width)
            y = np.linspace(0, 50., num=self.ov_env.cfg.img_height)
            X, Y = np.meshgrid(x, y)
            if self.stereo:
                titles = ["Left RGB", "Right RGB"]
            else:
                titles = ["RGB", "Depth"]

            # Set up the first depth map visualization
            self.rendered_img1 = self.ax1.imshow(np.zeros((self.ov_env.cfg.img_height, self.ov_env.cfg.img_width, 3)), vmin=0., vmax=1.)
            self.ax1.set_title(titles[0])

            # Set up the second depth map visualization
            if self.stereo:
                self.rendered_img2 = self.ax2.imshow(np.zeros((self.ov_env.cfg.img_height, self.ov_env.cfg.img_width, 3)), vmin=0., vmax=1.)
            else:
                self.rendered_img2 = self.ax2.imshow(X, vmin=0, vmax=1.4, cmap='Greys')
            self.ax2.set_title(titles[1])

            self.fig.canvas.draw()
            plt.show(block=False)

        if self.use_data_aug and self.img_aug_type == "rgb":
            self.rgb_aug = RgbAug(
                device=self.device,
                all_env_inds=self.ov_env.robot._ALL_INDICES,
                use_stereo=self.stereo,
                background_cfg={
                    "dir": os.path.join(
                        str(pathlib.Path(__file__).parent.parent.resolve()),
                        "assets", "background_imgs", "voc_resized"
                    ),
                    "height": self.ov_env.cfg.img_height,
                    "width": self.ov_env.cfg.img_width,
                    "aug_prob": 0.5
                },
                color_cfg={
                    "aug_prob": 1.,
                    "saturation_range": [0.5, 1.5],
                    "contrast_range": [0.5, 1.5],
                    "brightness_range": [0.5, 1.5],
                    "hue_range": [-0.15, 0.15]
                },
                motion_blur_cfg={
                    "aug_prob": 0.1,
                    "kernel_sizes": [9, 11, 13, 15, 17],
                    "angle_range": [0, 2*np.pi],
                    "direction_range": [-1, 1]
                }
            )
            voc_imgs_dir = os.path.join(
                str(pathlib.Path(__file__).parent.parent.resolve()),
                "assets", "background_imgs", "voc_resized"
            )
            voc_imgs = glob.glob(os.path.join(voc_imgs_dir, "*.jpg"))[:10]
            self.background_imgs = [
                Image.open(img_name).convert("RGB")
                for img_name in voc_imgs
            ]
            self.background_img_transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.init_tensors()

    def init_tensors(self):
        # dummy variable so that calculating neglogp doesn't give error (we don't care about the value)
        self.prev_actions_student = torch.zeros((self.num_envs, self.num_actions_student), dtype=torch.float32).to(self.device)
        self.prev_actions_teacher = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32).to(self.device)

        self.current_rewards = torch.zeros(
            (self.num_envs, self.value_size), dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones(
            (self.num_envs,), dtype=torch.uint8, device=self.device
        )

        self.actions_teacher = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32, device=self.device
        )

        if self.is_rnn:
            self.student_hidden_states = self.student_model.get_default_rnn_state()
            self.student_hidden_states = [s.to(self.device) for s in self.student_hidden_states]
            self.hidden_state_means = [
                RunningMeanStd((s.shape[0], s.shape[-1])).to(device=self.device, dtype=s.dtype)
                for s in self.student_hidden_states
            ]
            self.num_seqs = self.horizon_length // self.seq_length

        if self.is_teacher_rnn:
            self.teacher_hidden_states = self.teacher_model.get_default_rnn_state()
            self.teacher_hidden_states = [s.to(self.device) for s in self.teacher_hidden_states]
            # self.num_seqs = self.horizon_length // self.seq_length

        self.env_counter = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        if self.stereo:
            self.rgb_buffers_left = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
            self.rgb_buffers_right = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
            self.depth_buffers_left = torch.zeros(
                (self.num_envs, 1, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
            self.depth_buffers_right = torch.zeros(
                (self.num_envs, 1, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
        else:
            self.rgb_buffers = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
            self.depth_buffers = torch.zeros(
                (self.num_envs, 1, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)

    def distill(self):
        self.student_model.train()
        self.teacher_model.eval()
        # torch.set_float32_matmul_precision('high')

        obs = self.env.reset()[0]

        log_counter = 0
        total_loss = 0.

        self.optimizer.zero_grad()

        num_iters = self.num_iters
        num_iters_since_beta_dec = 0

        while log_counter < num_iters:
            if log_counter < 15_000:
                beta = 1.
            else:
                beta = 0.
                # if log_counter % 240 == 0 or num_iters_since_beta_dec > 1_000:
                #     num_iters_since_beta_dec = 0
                #     perf = self.ov_env.in_success_region.float().mean().cpu().numpy()
                #     if perf > 0.2:
                #         beta = max(beta - 0.05, 0.)
                #         print(f"Changing beta to new value: {beta}")
            beta = 0.
            if self.play_policy:
                beta = 0. if self.step_student_actions else 1.
                self.optimizer.param_groups[0]["lr"] = 0.0
            # beta = 0.

            if log_counter < 5000:
                self.finetune_backbone = False
            else:
                self.finetune_backbone = True

            if self.img_aug_type == "depth" and self.use_depth_aug:
                obs["img"] = self.augment_depth(obs["img"])
                obs["img"][obs["img"] > self.ov_env.cfg.d_max] = 0.
                obs["img"][obs["img"] < self.ov_env.cfg.d_min] = 0.

            if self.use_data_aug and self.img_aug_type == "rgb":
                if self.stereo:
                    imgs = {
                        "left_img": obs["img_left"],
                        "right_img": obs["img_right"]
                    }
                    masks = {
                        "left_mask": obs["mask_left"],
                        "right_mask": obs["mask_right"]
                    }
                    aug_output = self.rgb_aug.apply(imgs, masks)
                    obs["img_left"] = aug_output["left_img"]
                    obs["img_right"] = aug_output["right_img"]
                    obs['img_right'] = torch.flip(obs['img_right'], dims=(2,3))
                else:
                    if self.img_aug_type == "rgb":
                        obs["rgb"] = self.rgb_aug.apply(obs["rgb"], obs["mask"])
                        self.rgb_buffers[even_indices] = obs['rgb'][even_indices]
                        obs['rgb'] = self.rgb_buffers
            else:
                obs['img_right'] = torch.flip(obs['img_right'], dims=(2,3))


            if self.viz_imgs:
                if self.stereo:
                    obj_uv_left = obs["obj_uv_left"][2].clone().detach().cpu().numpy()
                    obj_uv_right = obs["obj_uv_right"][2].clone().detach().cpu().numpy()
                    obj_uv_left[0] *= self.ov_env.cfg.img_width
                    obj_uv_left[1] *= self.ov_env.cfg.img_height
                    obj_uv_right[0] *= self.ov_env.cfg.img_width
                    obj_uv_right[1] *= self.ov_env.cfg.img_height
                    # plot object uv on top of rgb, need to be int
                    obj_uv_left = obj_uv_left.astype(np.int32)
                    obj_uv_right = obj_uv_right.astype(np.int32)
                    rgb_img = obs["img_left"][2].clone().detach().cpu().numpy().transpose(1, 2, 0)
                    # rgb_img[obj_uv_left[1]-4:obj_uv_left[1]+4, obj_uv_left[0]-4:obj_uv_left[0]+4, :] = [1, 1, 1]
                    self.rendered_img1.set_data(rgb_img)
                    rgb_img = obs["img_right"][2].clone().detach().cpu().numpy().transpose(1, 2, 0)
                    # rgb_img[obj_uv_right[1]-4:obj_uv_right[1]+4, obj_uv_right[0]-4:obj_uv_right[0]+4, :] = [1, 1, 1]
                    self.rendered_img2.set_data(rgb_img)
                else:
                    rgb_img = obs["rgb"][0].clone().detach().cpu().numpy().transpose(1, 2, 0)
                    self.rendered_img1.set_data(rgb_img)
                    self.rendered_img2.set_data(obs["img"][0, 0].detach().cpu().numpy())
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            
            # left_img = (obs["img_left"][0].clone().detach().cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
            # right_img = (obs["img_right"][0].clone().detach().cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
            # Image.fromarray(left_img).save("left_img.png")
            # Image.fromarray(right_img).save("right_img.png")
            # breakpoint()
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    actions_teacher = self.get_actions(obs, "teacher")
                    self.actions_teacher = actions_teacher["actions"]

                start_time = time.time()
                # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # imgs_real = torch.load('images1.pth') #[-0.7, 0.08, 0.295]
                # imgs_real = torch.load('images2.pth') #[-0.65, 0.25,  0.3]
                # real_img_left = imgs_real['left_img']
                # real_img_right = imgs_real['right_img']
                # obs['img_left'][:] = real_img_left #[:, torch.arange(3 - 1, -1, -1), :, :]
                # obs['img_right'][:] = real_img_right #[:, torch.arange(3 - 1, -1, -1), :, :]
                actions_student = self.get_actions(obs, "student")

                aux_loss = list() if self.is_aux else [0.]
                if actions_student["aux"] is not None:
                    aux_out = actions_student["aux"]
                    self.aux_loss_names = aux_out.keys()
                    aux_gt = obs["aux_info"]
                    mask = obs["mask_left"] if self.stereo else obs["mask"]
                    # invert binary mask for depth
                    mask = ~mask
                    for aux_name in self.aux_loss_names:
                        num_vals = aux_out[aux_name].shape[-1]
                        if 'img' in aux_name:
                            num_supervised_envs = aux_out[aux_name].shape[0]
                            if "depth" in aux_name:
                                depth_min = self.ov_env.cfg.d_min
                                depth_max = self.ov_env.cfg.d_max
                                aux_out[aux_name] = aux_out[aux_name]*(depth_max - depth_min) + depth_min
                            aux_loss.append(
                                torch.mean(
                                    torch.norm(
                                        ((aux_out[aux_name] - aux_gt[aux_name])), 
                                        p=2, dim=(1,2,3)),
                                )
                            )
                            # breakpoint()
                            if self.rank == 0:
                                self.log_img(aux_out[aux_name][:5], aux_gt[aux_name][:5])
                        elif "uv" in aux_name:
                            # find uvs that are between 0 and 1
                            uv_mask = (aux_out[aux_name] >= 0) & (aux_out[aux_name] <= 1)
                            uv_mask = uv_mask.all(dim=-1)
                            aux_loss.append(
                                self.loss(
                                    aux_out[aux_name][uv_mask],
                                    aux_gt[aux_name][uv_mask].reshape(
                                        len(uv_mask), -1
                                    )
                                )
                            )
                        else:
                            aux_loss.append(
                                self.loss(aux_out[aux_name], aux_gt[aux_name].reshape(self.num_envs, -1)) #/ num_vals
                            )

                weights = 1 / actions_teacher['sigmas'][0]
                weights = weights ** 2
                student_loss = (
                    self.loss(
                        actions_student["mus"], actions_teacher["mus"],
                        fn="weighted_l2", weights=weights
                    ) +
                    self.loss(actions_student["sigmas"], actions_teacher["sigmas"])
                )
                total_loss += student_loss + self.aux_coeff*sum(aux_loss)
            # pos = torch.tensor([
            #     [self.ov_env.cfg.x_center+self.ov_env.cfg.x_width/2, self.ov_env.cfg.y_center+self.ov_env.cfg.y_width/2, 0.5],
            #     [self.ov_env.cfg.x_center-self.ov_env.cfg.x_width/2, self.ov_env.cfg.y_center-self.ov_env.cfg.y_width/2, 0.5],
            # ]).to(self.device)
            # self.ov_env._set_pos_marker(pos)
            # print(aux_out["object_pos"])
            # self.ov_env._set_pos_marker(aux_out["object_pos"])

            if self.rank == 0:
                self.log_information(log_counter, total_loss, aux_loss, beta)

            log_counter += 1
            self.env_counter += 1
            num_iters_since_beta_dec += 1

            if self.is_rnn:
                if log_counter % self.seq_length == 0:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), 1.0
                    )
                    self.optimizer.step()
                    # self.scheduler.step()
                    self.optimizer.zero_grad()
                    for i, s in enumerate(self.student_hidden_states):
                        self.student_hidden_states[i] = s.detach()
                    total_loss = 0.
                    torch.cuda.empty_cache()
            else:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                total_loss = 0.
            end_time = time.time()
            # print(f"Time taken for backward and step: {end_time - start_time} seconds")

            if beta is None:
                stepping_actions = actions_student["actions"] if self.step_student_actions else actions_teacher["actions"]
            else:
                p = torch.rand(self.num_envs) > beta
                stepping_actions = torch.zeros_like(actions_student["actions"])
                student_inds = p > beta
                teacher_inds = p <= beta
                if torch.any(student_inds):
                    stepping_actions[student_inds] = actions_student["actions"][student_inds]
                if torch.any(teacher_inds):
                    stepping_actions[teacher_inds] = actions_teacher["actions"][teacher_inds]

            obs, rew, out_of_reach, timed_out, info = self.env.step(
                stepping_actions.detach()
            )

            self.frame += self.num_envs
            self.current_rewards += rew.unsqueeze(-1)
            self.current_lengths += 1
            self.dones = out_of_reach | timed_out
            all_done_indices = self.dones.nonzero(as_tuple=False)

            if self.is_rnn and len(all_done_indices) > 0:
                if total_loss > 1e-8:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), 1.0
                    )
                    self.optimizer.step()
                    # self.scheduler.step()
                    self.optimizer.zero_grad()
                    for i, s in enumerate(self.student_hidden_states):
                        self.student_hidden_states[i] = s.detach()
                    total_loss = 0.

                for i, s in enumerate(self.student_hidden_states):
                    with torch.no_grad():
                        self.hidden_state_means[i](
                            self.student_hidden_states[i][:, all_done_indices[0]].permute((1, 0, 2))
                        )
                    self.student_hidden_states[i][:, all_done_indices] *= 0.

                self.env_counter[all_done_indices] = 0

            if self.is_teacher_rnn and len(all_done_indices) > 0:
                for s in self.teacher_hidden_states:
                    s[:, all_done_indices, ...] *= 0.

            done_indices = all_done_indices[:]
            if self.use_data_aug and self.img_aug_type == "rgb":
                self.rgb_aug.reset(done_indices)
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            self.actions_teacher[done_indices] *= 0.

            if not self.play_policy:
                # if (
                #     log_counter % 10000 == 0 and
                #     log_counter > 10 and
                #     self.optimizer.param_groups[0]["lr"] > 1.2*1e-4
                # ):
                #     self.optimizer.param_groups[0]["lr"] /= 1.2
                if self.rank == 0 and log_counter % 5_000 == 0:
                    ckpt_path = os.path.join(
                        self.nn_dir,
                        f"dextrah_student_{log_counter}_iters"
                    )
                    self.save(ckpt_path)

        if self.rank == 0 and self.use_wandb:
            wandb.finish()

    def cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, init_lr_frac= 0.01):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return init_lr_frac + (1 - init_lr_frac) * float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def augment_depth(self, depth_nchw):
        depth_nhw = depth_nchw[:, 0]
        depths = torch.clone(depth_nhw)
        self.depth_aug.add_correlated_noise(
            depth_nhw, depths, **self.depth_aug_cfg["correlated_noise"]
        )
        self.depth_aug.add_normal_noise(
            depths, **self.depth_aug_cfg["normal_noise"]
        )
        self.depth_aug.add_pixel_dropout_and_randu(
            depths, **self.depth_aug_cfg["pixel_dropout_and_randu"]
        )
        self.depth_aug.add_sticks(
            depths, **self.depth_aug_cfg["sticks"]
        )

        return depths.unsqueeze(1)

    def log_information(self, log_counter, total_loss, aux_loss=None, beta=None):
        student_loss = total_loss if aux_loss is None else total_loss - self.aux_coeff*sum(aux_loss)
        if beta is None:
            beta = 0.

        if self.game_rewards.current_size > 0:
            mean_rewards = self.game_rewards.get_mean()
            mean_lengths = self.game_lengths.get_mean()
            self.mean_rewards = mean_rewards[0]
            for i in range(self.value_size):
                rewards_name = "rewards" if i == 0 else "rewards{0}".format(i)
                self.writer.add_scalar(
                    rewards_name + "/step", mean_rewards[i], self.frame
                )
                self.writer.add_scalar(
                    "total_loss", total_loss.detach().cpu().numpy(), self.frame
                )
                self.writer.add_scalar(
                    "imitation_loss", student_loss.detach().cpu().numpy(), self.frame
                )
                self.writer.add_scalar(
                    "beta", beta, self.frame
                )
                if beta > 0.95:
                    perf = self.ov_env.in_success_region.float().mean().cpu().numpy()
                else:
                    perf = self.ov_env.in_success_region.float().mean().cpu().numpy()
                self.writer.add_scalar(
                    "in_success_region", perf, self.frame
                )
                if self.use_wandb:
                    wandb.log({
                        "in_success_region": perf,
                        "imitation_loss": student_loss.detach().cpu().numpy(),
                        "total_loss": total_loss.detach().cpu().numpy(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "beta": beta,
                        "iteration": self.frame
                    })
                if self.is_aux:
                    for idx, name in enumerate(self.aux_loss_names):
                        self.writer.add_scalar(
                            f"aux_loss_{name}", aux_loss[i].detach().cpu().numpy(), self.frame
                        )
                        if self.use_wandb:
                            wandb.log({
                                f"aux_loss_{name}": aux_loss[i].detach().cpu().numpy(),
                                "iteration": self.frame
                            })

        if log_counter % 10 == 0:
            print("="*10)
            print("ITERATION:", log_counter)
            print("LR: ", self.optimizer.param_groups[0]["lr"])
            print("Imitation Loss: ", student_loss)
            if self.is_aux:
                print("Aux Loss: ", aux_loss)
            print("Total Loss: ", total_loss)
            print("Beta: ", beta)
            if self.game_rewards.current_size > 0:
                print("\tMean Rewards: ", mean_rewards)
                print("\tMean Length: ", mean_lengths)
                print("\tin_success_region: ", perf)

    def log_img(self, pred_images, gt_images):
        combined_images = torch.cat((pred_images, gt_images), dim=0)
        image_grid = vutils.make_grid(combined_images, nrow=pred_images.shape[0], normalize=True, scale_each=True)
        self.writer.add_image('Predictions_vs_Ground_Truth', image_grid, global_step=self.frame)
        if self.use_wandb:
            images = wandb.Image(image_grid, caption="Top: Network Pred, Bottom: GT")
            wandb.log({"predictions vs ground truth": images})

    def get_actions(self, obs, policy_type):
        aux = None
        if policy_type == "student":
            # real_world_idx = 1
            # real_world_names = ["obs.pth", "obs2.pth"]
            # gt_pos = [[-0.7, 0.08, 0.295], [-0.65, 0.25, 0.3]]
            # real_obs = torch.load(real_world_names[real_world_idx])
            # gt_pos = torch.tensor(gt_pos[real_world_idx]).reshape(1, 3).to(self.device)
            # arm_positions = [
            #     -0.7875749180783324, -0.4724239581655329, 0.6733201341853008,
            #     1.211750626464511, -1.7481902752912126, 0.9306101177413156, 0.6660046636382199
            # ]
            # hand_positions = [
            #     -0.018107680695800783, 0.22323929877421064, 0.7489833809370932, 0.9548251041408286,
            #     -0.013491997381184897, 0.34635377487752284, 0.8157332627176921, 0.8537238869226075,
            #     -0.0008876314066569011, 0.4619233840242513, 0.8937560633628338, 0.8243432873622641,
            #     1.175845324398397, 0.3547862732407634, 0.3690771388879395, 0.286438654928182
            # ]
            # robot_q = torch.tensor(arm_positions + hand_positions).to(self.device)
            # obs[self.student_obs_type][:, :len(robot_q)] = robot_q
            batch_dict = {
                "is_train": True,
                # "obs": real_obs["proprio"].to(self.device).repeat(2,1),
                "obs": obs[self.student_obs_type],
                # "observations": obs[self.student_obs_type],
                "prev_actions": self.prev_actions_student,
            }
            if "img" in obs:
                # mean_tensor = torch.mean(obs["img"], dim=(2, 3), keepdim=True)
                batch_dict["img"] = obs["img"] #- mean_tensor
                batch_dict["rgb_data"] = obs["rgb"]
                batch_dict["rgb"] = obs["rgb"]
            if "img_left" in obs:
                batch_dict["img_left"] = obs["img_left"]
                batch_dict["img_right"] = obs["img_right"]
                # batch_dict["img_left"] = real_obs["left_img"].repeat(2,1,1,1).to(self.device)
                # batch_dict["img_right"] = real_obs["right_img"].repeat(2,1,1,1).to(self.device)
            if self.is_rnn:
                # batch_dict["rnn_states"] = [real_obs["hidden_state_1"], real_obs["hidden_state_2"]]
                batch_dict["rnn_states"] = self.student_hidden_states
                batch_dict["seq_length"] = 1
                batch_dict["rnn_masks"] = None
            batch_dict["finetune_backbone"] = self.finetune_backbone
            res_dict = self.student_model_ddp(batch_dict)
            mus = res_dict["mus"]
            sigmas = res_dict["sigmas"]
            # self.ov_env._set_gt_pos_marker(gt_pos.repeat(self.num_envs, 1))
            # breakpoint()
            if self.is_rnn:
                if self.is_aux:
                    self.student_hidden_states = [s for s in res_dict["rnn_states"][0]]
                else:
                    self.student_hidden_states = [s for s in res_dict["rnn_states"]]
            if self.is_aux:
                aux = res_dict["rnn_states"][1]
        else:
            batch_dict = {
                "is_train": False,
                "obs": obs[self.teacher_obs_type],
                "prev_actions": self.prev_actions_teacher,
            }
            if self.is_teacher_rnn:
                batch_dict["rnn_states"] = self.teacher_hidden_states
                batch_dict["seq_length"] = 1
                batch_dict["rnn_masks"] = None
            res_dict = self.teacher_model(batch_dict)
            if self.is_teacher_rnn:
                self.teacher_hidden_states = res_dict["rnn_states"]
            mus = res_dict["mus"]
            sigmas = res_dict["sigmas"]
        distr = torch.distributions.Normal(mus, sigmas, validate_args=False)
        selected_action = distr.sample().squeeze()
        # clamp selected action between 1 and -1
        selected_action = torch.clamp(selected_action, -1., 1.)

        return {
            "mus": mus,
            "sigmas": sigmas,
            "actions": selected_action,
            "aux": aux
        }

    def loss(self, student_result, target_result, fn="l2", weights=None):
        if fn == "l2":
            loss = l2(student_result, target_result)
        else:
            loss = weighted_l2(student_result, target_result, weights)
        rnn_masks = None
        losses, sum_mask = torch_ext.apply_masks(
            [loss.unsqueeze(1)], rnn_masks
        )
        return losses[0]

    def set_weights(self, ckpt, policy_type):
        """Set the weights of the model."""
        weights = torch_ext.load_checkpoint(ckpt)
        if policy_type == "student":
            weights["model"] = adjust_state_dict_keys(
                weights["model"],
                self.student_model.state_dict()
            )
            model = self.student_model
            # self.epoch_num = weights.get('epoch', 0)
            # self.optimizer.load_state_dict(weights['optimizer'])
            # self.frame = weights.get('frame', 0)
        else:
            model = self.teacher_model
        model.load_state_dict(weights["model"])
        if self.normalize_input and 'running_mean_std' in weights:
            model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def save(self, filename):
        """Save the checkpoint to filename"""
        state = {
            "model": self.student_model.state_dict()
        }
        state['epoch'] = self.epoch_num
        state['optimizer'] = self.optimizer.state_dict()
        state['frame'] = self.frame
        torch_ext.save_checkpoint(filename, state)

    def load_networks(self, params):
        """Loads the network """
        builder = ModelBuilder()
        return builder.load(params)

    def load_param_dict(self, cfg_path) -> Dict:
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

