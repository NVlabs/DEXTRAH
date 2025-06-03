"""Script to evaluate a student policy checkpoint.

This script provides functionality to evaluate a trained student policy in the Dextrah environment.
It supports both mono and stereo vision models, with options for data recording and video creation.
Please check the README for example usage.

Data Recording:
    When --record_data is enabled, the script will:
    - Create a timestamped directory for each run
    - Save data in HDF5 format with associated YAML metadata
    - Optionally create MP4 videos from recorded frames
    - Disable environment randomization for consistent recording
    - Record observations, actions, and object positions

Output Structure:
    save_dir/
    └── recorded_data_YYYYMMDD_HHMMSS/
        ├── data/
        │   ├── data_0.h5
        │   ├── data_0.yaml
        │   └── ...
        └── videos/
            ├── env_0_file_0.mp4
            └── ...
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a student policy checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the student policy checkpoint")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
parser.add_argument("--save_dir", type=str, default="eval_results", help="Directory to save evaluation results")
parser.add_argument("--record_data", action="store_true", default=False, help="Record data during evaluation")
parser.add_argument("--max_records_per_file", type=int, default=100, help="Maximum number of steps to record in a single file")
parser.add_argument("--create_video", action="store_true", default=False, help="Create videos for the recorded data")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import pathlib
import torch
import numpy as np
from datetime import datetime
import yaml
import gymnasium as gym
import glob
from torchvision import transforms
from PIL import Image

from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_builder import ModelBuilder

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import the task setup
import dextrah_lab.tasks.dextrah_kuka_allegro.gym_setup  # noqa: F401

from dextrah_lab.distillation.a2c_with_aux_depth import A2CBuilder as A2CWithAuxDepthBuilder
from dextrah_lab.distillation.a2c_with_aux_cnn import A2CBuilder as A2CWithAuxCNNBuilder
from dextrah_lab.distillation.a2c_with_aux_cnn_stereo import A2CBuilder as A2CWithAuxCNNStereoBuilder
from dextrah_lab.distillation.a2c_with_aux_cnn_stereo_recon import A2CBuilder as A2CWithAuxCNNStereoReconBuilder
from dextrah_lab.distillation.a2c_with_pretrain import A2CBuilder as A2CWithPretrainBuilder
from dextrah_lab.distillation.a2c_stereo_transformer import A2CBuilder as A2CStereoTransformerBuilder
from dextrah_lab.distillation.a2c_mono_resnet import A2CBuilder as A2CMonoResnetBuilder
from dextrah_lab.distillation.a2c_mono_transformer import A2CBuilder as A2CMonoTransformerBuilder

from rgb_augs import RgbAug

from dextrah_lab.distillation.data_recorder import DataRecorder

def adjust_state_dict_keys(checkpoint_state_dict, model_state_dict):
    """Adjust state dict keys to match model architecture."""
    adjusted_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            adjusted_state_dict[key] = value
        else:
            parts = key.split(".")
            new_key_with_orig_mod = None
            parts.insert(2, "_orig_mod")

            new_key_with_orig_mod = ".".join(parts)
            if new_key_with_orig_mod in model_state_dict:
                adjusted_state_dict[new_key_with_orig_mod] = value
            else:
                key_no_orig_mod = key.replace("_orig_mod.", "")
                if key_no_orig_mod in model_state_dict:
                    adjusted_state_dict[key_no_orig_mod] = value
                else:
                    print(f"Could not match key: {key} -> {new_key_with_orig_mod}")
                    adjusted_state_dict[key] = value
    return adjusted_state_dict

class PolicyEvaluator:
    def __init__(self, env, config, checkpoint_path, save_dir=None, record_data=False, max_records_per_file=10, create_video=True):
        self.env = env
        self.ov_env = env.env
        self.num_envs = self.ov_env.num_envs
        self.num_actions = self.ov_env.num_actions
        self.device = torch.device("cuda:0")
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.record_data = record_data
        
        # Initialize video recorder if needed
        if self.record_data and save_dir is not None:
            self.data_recorder = DataRecorder(
                save_dir=save_dir,
                num_envs=self.num_envs,
                img_height=self.ov_env.cfg.img_height,
                img_width=self.ov_env.cfg.img_width,
                stereo=self.ov_env.simulate_stereo,
                max_records_per_file=max_records_per_file,
                create_video=create_video
            )
        else:
            self.data_recorder = None
        
        # Load network parameters
        self.network_params = self.load_param_dict(self.config["student"]["cfg"])["params"]
        self.network = self.load_networks(self.network_params)
        
        # Model configuration
        self.value_size = 1
        self.horizon_length = self.network_params["config"]["horizon_length"]
        self.normalize_value = self.network_params["config"]["normalize_value"]
        self.normalize_input = self.network_params["config"]["normalize_input"]
        
        # Initialize model
        self.model_config = {
            "actions_num": self.num_actions,
            "input_shape": (self.ov_env.num_observations,),
            "batch_size": self.num_envs,
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        
        self.model = self.network.build(self.model_config).to(self.device)
        self.load_checkpoint(checkpoint_path)
        
        # Check for auxiliary tasks
        if hasattr(self.model.a2c_network, "is_aux") and self.model.a2c_network.is_aux:
            self.is_aux = True
            print("USING AUX")
        else:
            self.is_aux = False
        
        # Initialize tensors
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32).to(self.device)
        self.is_rnn = self.model.is_rnn()
        if self.is_rnn:
            # Set sequence length to 1 for evaluation, matching distillation.py
            self.seq_length = 1
            # Initialize RNN states as a tuple of tensors
            self.hidden_states = tuple(s.to(self.device) for s in self.model.get_default_rnn_state())
        
        # Initialize metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
        # Initialize observation buffers
        if self.ov_env.simulate_stereo:
            self.rgb_buffers_left = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
            self.rgb_buffers_right = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
        else:
            self.rgb_buffers = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width)
            ).to(self.device)
            
        # Initialize RGB augmentation if needed
        if self.ov_env.cfg.img_aug_type == "rgb":
            self._setup_rgb_augmentation()
            
    def _setup_rgb_augmentation(self):
        """Initialize RGB augmentation with zero probabilities for evaluation."""
        # Set all augmentation probabilities to 0 for evaluation
        rgb_aug_prob = 0.0
        color_aug_prob = 0.0
        motion_blur_prob = 0.0
        
        self.rgb_aug = RgbAug(
            device=self.device,
            all_env_inds=self.ov_env.robot._ALL_INDICES,
            use_stereo=self.ov_env.simulate_stereo,
            background_cfg={
                "dir": os.path.join(
                    str(pathlib.Path(__file__).parent.parent.resolve()),
                    "assets", "background_imgs", "voc_resized"
                ),
                "height": self.ov_env.cfg.img_height,
                "width": self.ov_env.cfg.img_width,
                "aug_prob": rgb_aug_prob
            },
            color_cfg={
                "aug_prob": color_aug_prob,
                "saturation_range": [0.5, 1.5],
                "contrast_range": [0.5, 1.5],
                "brightness_range": [0.5, 1.5],
                "hue_range": [-0.15, 0.15]
            },
            motion_blur_cfg={
                "aug_prob": motion_blur_prob,
                "kernel_sizes": [9, 11, 13, 15, 17],
                "angle_range": [0, 2*np.pi],
                "direction_range": [-1, 1]
            }
        )
        
        # Load background images (needed for augmentation structure)
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

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint."""
        weights = torch.load(checkpoint_path, map_location=self.device)
        if "model" in weights:
            weights["model"] = adjust_state_dict_keys(
                weights["model"],
                self.model.state_dict()
            )
            self.model.load_state_dict(weights["model"])
        else:
            self.model.load_state_dict(weights)
            
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])
            
    def get_actions(self, obs):
        """Get actions from the policy."""
        # Handle image observations
        even_indices = torch.where(self.env_counter % 2 == 0)[0]
        
        if self.ov_env.simulate_stereo:
            # Handle stereo images exactly as in distillation.py
            imgs = {
                "left_img": obs["img_left"],
                "right_img": obs["img_right"]
            }
            masks = {
                "left_mask": obs["mask_left"],
                "right_mask": obs["mask_right"]
            }
            # Apply RGB augmentation (will do nothing since probabilities are 0)
            aug_output = self.rgb_aug.apply(imgs, masks)
            obs["img_left"] = aug_output["left_img"]
            obs["img_right"] = aug_output["right_img"]
            
            self.rgb_buffers_left[even_indices] = obs['img_left'][even_indices]
            obs['img_left'] = self.rgb_buffers_left
            obs['img_right'] = torch.flip(obs['img_right'], dims=(2,3))  # Flip right image
            self.rgb_buffers_right[even_indices] = obs['img_right'][even_indices]
            obs['img_right'] = self.rgb_buffers_right
        else:
            if self.ov_env.cfg.img_aug_type == "rgb":
                # Apply RGB augmentation (will do nothing since probabilities are 0)
                obs["rgb"] = self.rgb_aug.apply(obs["rgb"], obs["mask"])
                self.rgb_buffers[even_indices] = obs['rgb'][even_indices]
                obs['rgb'] = self.rgb_buffers
        
        batch_dict = {
            "is_train": False,  # During evaluation, we don't need gradients
            "obs": obs[self.config["student"]["obs_type"]],
            "prev_actions": self.prev_actions,
        }
        
        # Add image observations if they exist
        if "img" in obs:
            batch_dict["img"] = obs["img"]
            batch_dict["rgb_data"] = obs["rgb"]
            batch_dict["rgb"] = obs["rgb"]
            
        # Add stereo image observations if they exist
        if "img_left" in obs:
            batch_dict["img_left"] = obs["img_left"]
            batch_dict["img_right"] = obs["img_right"]
            
        if self.is_rnn:
            batch_dict["rnn_states"] = self.hidden_states
            batch_dict["seq_length"] = self.seq_length
            batch_dict["rnn_masks"] = None
            
        # Add finetune_backbone flag (always False during evaluation)
        batch_dict["finetune_backbone"] = False
            
        res_dict = self.model(batch_dict)
        
        if self.is_rnn:
            # Handle RNN states - convert to tuple for PyTorch RNN
            if self.is_aux:
                self.hidden_states = tuple(s.detach() for s in res_dict["rnn_states"][0])
            else:
                self.hidden_states = tuple(s.detach() for s in res_dict["rnn_states"])
        
        mus = res_dict["mus"]
        sigmas = res_dict["sigmas"]
        distr = torch.distributions.Normal(mus, sigmas, validate_args=False)
        selected_action = distr.sample().squeeze()
        selected_action = torch.clamp(selected_action, -1., 1.)
        
        # Store actions for next step
        self.prev_actions = selected_action.detach()
        
        return selected_action.detach()  # Ensure actions are detached before stepping
        
    def evaluate(self, num_episodes):
        """Evaluate the policy for specified number of episodes."""
        self.model.eval()
        
        for episode in range(num_episodes):
            obs = self.env.reset()[0]
            episode_reward = 0
            episode_length = 0
            dones = torch.zeros((self.num_envs,), dtype=torch.uint8, device=self.device)
            self.env_counter = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
            
            # Reset RNN states at the start of each episode
            if self.is_rnn:
                self.hidden_states = tuple(s.to(self.device) for s in self.model.get_default_rnn_state())
            
            while not dones.all():
                with torch.no_grad():
                    action = self.get_actions(obs)
                
                obs, reward, out_of_reach, timed_out, info = self.env.step(action)
                dones = out_of_reach | timed_out
                
                # Record step if video recording is enabled
                if self.record_data and self.data_recorder is not None:
                    self.data_recorder.record_step(obs, action, reward, dones, self.prev_actions)
                
                # Get indices of done environments
                all_done_indices = dones.nonzero(as_tuple=False)
                
                # Accumulate rewards only for non-done environments
                not_dones = 1.0 - dones.float()
                episode_reward += (reward * not_dones).mean().item()
                episode_length += 1
                self.env_counter += 1
                
                # Handle RNN states for done environments
                if self.is_rnn and len(all_done_indices) > 0:
                    hidden_states_list = list(self.hidden_states)
                    for s in hidden_states_list:
                        s[:, all_done_indices] *= 0.
                    self.hidden_states = tuple(hidden_states_list)
                
                # Reset environment counter for done environments
                if len(all_done_indices) > 0:
                    self.env_counter[all_done_indices] = 0
                    
                    # Reset image buffers for done environments if using RGB augmentation
                    if hasattr(self, 'rgb_buffers'):
                        if self.ov_env.simulate_stereo:
                            self.rgb_buffers_left[all_done_indices] = 0
                            self.rgb_buffers_right[all_done_indices] = 0
                        else:
                            self.rgb_buffers[all_done_indices] = 0
                            
                    # Reset RGB augmentation for done environments
                    if self.ov_env.cfg.img_aug_type == "rgb":
                        done_indices = all_done_indices[:]
                        self.rgb_aug.reset(done_indices)
            
            # Save any remaining recorded data at the end of episode
            if self.record_data and self.data_recorder is not None and self.data_recorder.recording_step_counter > 0:
                self.data_recorder.save_and_create_videos()
            
            # Get final success rate from the environment
            success_rate = self.ov_env.in_success_region.float().mean().item()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_rates.append(success_rate)
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Length: {episode_length}")
            print(f"Success: {success_rate:.2f}")
            print("-" * 50)
            
        # Print final statistics
        print("\nEvaluation Results:")
        print(f"Average Reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
        print(f"Average Length: {np.mean(self.episode_lengths):.2f} ± {np.std(self.episode_lengths):.2f}")
        print(f"Success Rate: {np.mean(self.success_rates):.2f} ± {np.std(self.success_rates):.2f}")
        
    def load_networks(self, params):
        """Load network builder."""
        builder = ModelBuilder()
        return builder.load(params)
        
    def load_param_dict(self, cfg_path):
        """Load network parameters from config file."""
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Main evaluation function."""
    # Set up environment
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        
    # Disable reset and randomization for data recording
    if args_cli.record_data:
      env_cfg.disable_out_of_reach_done = True
      env_cfg.disable_arm_randomization = True
      env_cfg.disable_dome_light_randomization = True

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Set up student configuration
    parent_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
    agent_cfg_folder = "dextrah_lab/tasks/dextrah_kuka_allegro/agents"
    
    if env.env.simulate_stereo:
        student_cfg = os.path.join(
            parent_path,
            agent_cfg_folder,
            "rl_games_ppo_stereo_transformer.yaml"
        )
    else:
        student_cfg = os.path.join(
            parent_path,
            agent_cfg_folder,
            "rl_games_ppo_mono_transformer.yaml"
        )
    
    # Register network builders
    model_builder.register_network("a2c_aux_depth_enc", A2CWithAuxDepthBuilder)
    model_builder.register_network("a2c_aux_cnn_net", A2CWithAuxCNNBuilder)
    model_builder.register_network("a2c_aux_cnn_net_stereo", A2CWithAuxCNNStereoBuilder)
    model_builder.register_network("a2c_aux_cnn_net_stereo_recon", A2CWithAuxCNNStereoReconBuilder)
    model_builder.register_network("a2c_aux_pretrain", A2CWithPretrainBuilder)
    model_builder.register_network("a2c_stereo_transformer", A2CStereoTransformerBuilder)
    model_builder.register_network("a2c_mono_resnet", A2CMonoResnetBuilder)
    model_builder.register_network("a2c_mono_transformer", A2CMonoTransformerBuilder)
    
    # Create evaluator
    config = {
        "student": {
            "cfg": student_cfg,
            "obs_type": "policy",
        }
    }
    
    evaluator = PolicyEvaluator(
        env=env,
        config=config,
        checkpoint_path=args_cli.checkpoint,
        save_dir=args_cli.save_dir,
        record_data=args_cli.record_data,
        max_records_per_file=args_cli.max_records_per_file,
        create_video=args_cli.create_video  # Use the video flag to control video creation
    )
    
    # Create save directory
    os.makedirs(args_cli.save_dir, exist_ok=True)
    
    # Run evaluation
    evaluator.evaluate(args_cli.num_episodes)
    
    # Save results
    results = {
        "episode_rewards": evaluator.episode_rewards,
        "episode_lengths": evaluator.episode_lengths,
        "success_rates": evaluator.success_rates,
        "mean_reward": np.mean(evaluator.episode_rewards),
        "std_reward": np.std(evaluator.episode_rewards),
        "mean_length": np.mean(evaluator.episode_lengths),
        "std_length": np.std(evaluator.episode_lengths),
        "mean_success": np.mean(evaluator.success_rates),
        "std_success": np.std(evaluator.success_rates),
    }
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args_cli.save_dir, f"eval_results_{timestamp}.npy")
    np.save(results_file, results)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
