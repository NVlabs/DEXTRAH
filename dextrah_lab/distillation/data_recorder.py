"""Data recording functionality for Dextrah environment.

This module provides functionality to record and save environment data and videos.
It supports both mono and stereo vision models, with options for data recording and video creation.
"""

import os
import torch
import numpy as np
from datetime import datetime
import yaml
import h5py
import cv2

class DataRecorder:
    """Handles recording and saving of environment data and videos."""
    
    def __init__(self, save_dir, num_envs, img_height, img_width, stereo=False, max_records_per_file=10, create_video=True):
        """Initialize the video recorder.
        
        Args:
            save_dir (str): Directory to save recorded data and videos
            num_envs (int): Number of environments
            img_height (int): Height of the images
            img_width (int): Width of the images
            stereo (bool): Whether to record stereo images
            max_records_per_file (int): Maximum number of steps to record in a single file
            create_video (bool): Whether to create MP4 videos from recorded data
        """
        self.save_dir = save_dir
        self.num_envs = num_envs
        self.img_height = img_height
        self.img_width = img_width
        self.stereo = stereo
        self.file_counter = 0
        self.max_records_per_file = max_records_per_file
        self.recording_step_counter = 0
        self.create_video = create_video
        
        # Create timestamped directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(save_dir, f'recorded_data_{timestamp}')
        
        # Create subdirectories
        self.data_dir = os.path.join(self.run_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Only create video directory if video creation is enabled
        if self.create_video:
            self.video_dir = os.path.join(self.run_dir, 'videos')
            os.makedirs(self.video_dir, exist_ok=True)
        
        # Initialize data storage
        self.reset_data_storage()
        
    def reset_data_storage(self):
        """Reset the data storage for a new recording session."""
        self.data_to_record = {
            'image_left': [],
            'image_right': [],
            'obs': [],
            'prev_actions': [],
            'actions': [],
            'object_pos': []
        }
        
    def record_step(self, obs, action, reward, done, prev_actions):
        """Record a single step of data.
        
        Args:
            obs (dict): Observation dictionary containing images
            action (torch.Tensor): Actions taken
            reward (torch.Tensor): Rewards received
            done (torch.Tensor): Done flags
            prev_actions (torch.Tensor): Previous actions from the policy
        """
        # Record images
        self.data_to_record['image_left'].append(obs['img_left'].detach().cpu())
        if self.stereo:
            self.data_to_record['image_right'].append(obs['img_right'].detach().cpu())
            
        # Record other data
        self.data_to_record['obs'].append(obs['policy'].detach().cpu())
        self.data_to_record['prev_actions'].append(prev_actions.detach().cpu())
        self.data_to_record['actions'].append(action.detach().cpu())
        self.data_to_record['object_pos'].append(obs['aux_info']['object_pos'].detach().cpu())
        
        self.recording_step_counter += 1
        
        # Save to h5py file when we reach max_records_per_file
        if self.recording_step_counter >= self.max_records_per_file:
            self.save_and_create_videos()
            self.recording_step_counter = 0
        
    def save_and_create_videos(self):
        """Save recorded data and create videos."""
        if len(self.data_to_record['image_left']) == 0:
            print("No data to save")
            return
            
        # Save data to h5 file
        self._save_recorded_data()
        
        # Create videos only if enabled
        if self.create_video:
            self._create_environment_videos()
        
        # Reset storage for next recording
        self.reset_data_storage()
        self.file_counter += 1
        
    def _save_recorded_data(self):
        """Save recorded data to h5py file and metadata to YAML."""
        # Create metadata dictionary
        metadata = {
            'timestamp': str(datetime.now()),
            'steps_recorded': len(self.data_to_record['image_left']),
            'stereo': self.stereo,
            'num_envs': self.num_envs,
            'img_height': self.img_height,
            'img_width': self.img_width,
            'data_shapes': {
                k: [len(v)] + (list(v[0].shape) if len(v) > 0 else []) 
                for k, v in self.data_to_record.items() if v is not None
            }
        }
        
        # Base filename without extension
        base_filename = os.path.join(self.data_dir, f'data_{self.file_counter}')
        h5_filename = f"{base_filename}.h5"
        yaml_filename = f"{base_filename}.yaml"
        
        # Save metadata to YAML file
        with open(yaml_filename, 'w') as yaml_file:
            yaml.dump(metadata, yaml_file, default_flow_style=False)
        print(f"Saved metadata to {yaml_filename}")
        
        # Save data to h5 file
        print(f"Saving recorded data to {h5_filename}")
        with h5py.File(h5_filename, 'w') as f:
            # Save metadata as attributes
            for key, value in metadata.items():
                if key != 'data_shapes':
                    f.attrs[key] = value
            
            # Save recorded data
            for k, v in self.data_to_record.items():
                if v is not None and len(v) > 0:
                    if 'image' in k:
                        # Stack the images and convert to float32
                        stacked_data = torch.stack(v, dim=0).to(torch.float32)
                        
                        # Handle dimensions
                        if len(stacked_data.shape) == 5:  # [T, B, C, H, W]
                            T, B, C, H, W = stacked_data.shape
                            stacked_data = stacked_data.reshape(T*B, C, H, W)
                            stacked_data = stacked_data.permute(0, 2, 3, 1).contiguous()
                        elif len(stacked_data.shape) == 4:  # [B, C, H, W]
                            stacked_data = stacked_data.permute(0, 2, 3, 1).contiguous()
                        
                        print(f"Saving {k} with shape {stacked_data.shape}")
                        f.create_dataset(k, data=stacked_data.cpu().numpy(), compression="gzip", compression_opts=3)
                    else:
                        # For non-image data
                        stacked_data = torch.stack(v, dim=0).to(torch.float32)
                        
                        if len(stacked_data.shape) > 2:
                            orig_shape = stacked_data.shape
                            stacked_data = stacked_data.reshape(orig_shape[0], -1)
                            print(f"Reshaping {k} from {orig_shape} to {stacked_data.shape}")
                        
                        print(f"Saving {k} with shape {stacked_data.shape}")
                        f.create_dataset(k, data=stacked_data.cpu().numpy(), compression="gzip", compression_opts=3)
        
        print(f"Saved {len(self.data_to_record['image_left'])} steps of data to {h5_filename}")
        
    def _create_environment_videos(self):
        """Create MP4 videos for each environment using the saved images."""
        h5_filename = os.path.join(self.data_dir, f'data_{self.file_counter}.h5')
        
        print(f"Creating videos from data in {h5_filename}")
        
        try:
            with h5py.File(h5_filename, 'r') as f:
                # Get metadata
                num_envs = f.attrs.get('num_envs', 1)
                steps_recorded = f.attrs.get('steps_recorded', 0)
                
                # Get images
                if 'image_left' in f:
                    all_images = f['image_left'][...]
                    
                    if len(all_images) == 0:
                        print("No frames found in the h5 file.")
                        return
                    
                    # Create a video for each environment
                    for env_id in range(num_envs):
                        # Extract frames for this environment
                        env_frames = []
                        for i in range(steps_recorded):
                            if i * num_envs + env_id < len(all_images):
                                frame = all_images[i * num_envs + env_id]
                                
                                # Convert to uint8 format for OpenCV
                                if frame.shape[-1] == 3:  # RGB
                                    frame = np.clip(frame, 0, 1) * 255
                                    frame = frame.astype(np.uint8)
                                else:  # Grayscale/depth
                                    frame = frame[:, :, 0]
                                    if np.max(frame) > 0:
                                        frame = (frame / np.max(frame)) * 255
                                    frame = frame.astype(np.uint8)
                                    frame = np.stack([frame, frame, frame], axis=-1)
                                
                                env_frames.append(frame)
                        
                        if not env_frames:
                            print(f"No frames found for environment {env_id}")
                            continue
                        
                        # Create video
                        video_filename = os.path.join(self.video_dir, f'env_{env_id}_file_{self.file_counter}.mp4')
                        frame_height, frame_width = env_frames[0].shape[:2]
                        
                        # Initialize video writer
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = 10
                        video_writer = cv2.VideoWriter(
                            video_filename, fourcc, fps, (frame_width, frame_height)
                        )
                        
                        for frame in env_frames:
                            video_writer.write(frame)
                        
                        video_writer.release()
                        print(f"Created video for environment {env_id}: {video_filename}")
                else:
                    print("No 'image_left' dataset found in the h5 file")
        except Exception as e:
            print(f"Error creating videos: {e}") 