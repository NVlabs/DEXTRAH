# System imports
import os
import sys
import pathlib
from threading import Lock, Thread
import copy
import argparse

# ROS imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import JointState, Image
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Bool

# Third party
import torch
import yaml
import time
import numpy as np

# CV
from cv_bridge import CvBridge
import cv2

# RL games
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.algos_torch import torch_ext

# Fabrics
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.utils.utils import initialize_warp

from dextrah_lab.tasks.dextrah_kuka_allegro.dextrah_kuka_allegro_utils import compute_absolute_action, assert_equals
from dextrah_lab.tasks.dextrah_kuka_allegro.dextrah_kuka_allegro_constants import (
    NUM_XYZ,
    NUM_RPY,
    NUM_HAND_PCA,
    HAND_PCA_MINS,
    HAND_PCA_MAXS,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,

)

# Dextrah FGP
from dextrah_lab.distillation.a2c_with_aux_cnn_stereo import A2CBuilder as A2CWithAuxCNNStereoBuilder
from dextrah_lab.distillation.a2c_stereo_transformer import A2CBuilder as A2CStereoTransformerBuilder

from dextrah_lab.deployment_scripts.policy_inference_stereo import RLGamesPolicy as DextrAHFGP

#cv2.namedWindow('Left RGB Image', cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow('Right RGB Image', cv2.WINDOW_AUTOSIZE)

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

#    adjusted_state_dict = {}
#
#    for key, value in checkpoint_state_dict.items():
#        # If the key is in the model's state_dict, use it directly
#        if key in model_state_dict:
#            adjusted_state_dict[key] = value
#        else:
#            # Try inserting '_orig_mod' in different positions based on key structure
#            parts = key.split(".")
#            new_key_with_orig_mod = None
#
#            # Try inserting '_orig_mod' before the last layer index for different cases
#            parts.insert(2, "_orig_mod")
#            new_key_with_orig_mod = ".".join(parts)
#
#            # If adding '_orig_mod' matches a key in the model, use the modified key
#            if new_key_with_orig_mod in model_state_dict:
#                adjusted_state_dict[new_key_with_orig_mod] = value
#            else:
#                # Log the key that couldn't be matched, for debugging purposes
#                print(f"Could not match key: {key} -> {new_key_with_orig_mod}")
#                # If neither works, retain the original key as a fallback
#                adjusted_state_dict[key] = value
#
#    return adjusted_state_dict

class DextrahFGP:
    def __init__(
        self, cfg_path, img_shape, num_proprio_obs,
        num_actions, ckpt_path, device
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

        if self.model.is_rnn():
            hidden_states = self.model.get_default_rnn_state()
            self.hidden_states = [
                s.to(self.device) for s in hidden_states
            ]

        # dummy variable, this doesn't actually contain prev actions
        # need this bc of rl_games weirdness...
        self.dummy_prev_actions = torch.zeros(
            (2, num_actions), dtype=torch.float32
        ).to(self.device)

    def step(self, proprio, left_image, right_image, hidden_states=None):
        batch_dict = {
            "is_train": True,
            "obs": proprio.repeat(2, 1),
            "img_left": left_image.repeat(2, 1, 1, 1),
            "img_right": right_image.repeat(2, 1, 1, 1),
            "prev_actions": self.dummy_prev_actions,
        }
        # add extra information for RNNs
        if self.model.is_rnn():
            batch_dict["rnn_states"] = self.hidden_states
            batch_dict["seq_length"] = 1
            batch_dict["rnn_masks"] = None

        # step through model
        res_dict = self.model(batch_dict)
        mus = res_dict["mus"][0:1]
        sigmas = res_dict["sigmas"][0:1]

        self.hidden_states = [
            s for s in res_dict["rnn_states"][0]
        ]

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
            self.step(
                dummy_proprio, dummy_left_img, dummy_right_img
            )

        self.reset_hidden_state()
        self.cuda_graph = torch.cuda.CUDAGraph()

        self.static_proprio = torch.empty_like(dummy_proprio, device=self.device)
        self.static_left_img = torch.empty_like(dummy_left_img, device=self.device)
        self.static_right_img = torch.empty_like(dummy_right_img, device=self.device)
        self.hidden_state_1 = torch.empty_like(self.hidden_states[0], device=self.device)
        self.hidden_state_2 = torch.empty_like(self.hidden_states[1], device=self.device)

        with torch.cuda.graph(self.cuda_graph):
            self._policy_out = self.step(
                self.static_proprio,
                self.static_left_img,
                self.static_right_img,
                [self.hidden_state_1, self.hidden_state_2]
            )
        
    def step_cuda_graph(self, proprio, left_img, right_img, save):
        self.static_proprio.copy_(proprio)
        self.static_left_img.copy_(left_img)
        self.static_right_img.copy_(right_img)
        self.hidden_state_1.copy_(self.hidden_states[0])
        self.hidden_state_2.copy_(self.hidden_states[1])
        self.cuda_graph.replay()

        policy_out = self._policy_out
        policy_out = {
            "mus": self._policy_out["mus"].clone(),
            "sigmas": self._policy_out["sigmas"].clone(),
            "obj_pos": self._policy_out["obj_pos"].clone(),
        }

        if save:
            obs_action_dict = {'proprio': proprio,
                        'left_img': left_img,
                        'right_img': right_img,
                        'hidden_state_1': self.hidden_states[0],
                        'hidden_state_2': self.hidden_states[1],
                        "mus": self._policy_out["mus"].clone(),
                        "sigmas": self._policy_out["sigmas"].clone(),
                        "obj_pos": self._policy_out["obj_pos"].clone(),
                        }

            #torch.save(obs_action_dict, 'obs.pth')
            #sys.exit()

        policy_out["selected_action"] = torch.distributions.Normal(
            policy_out["mus"],
            policy_out["sigmas"]
        ).sample()
    
        return policy_out

    def reset_hidden_state(self):
        for i in range(len(self.hidden_states)):
            self.hidden_states[i].zero_() # = self.hidden_states[i] * 0.

class DextrahFGPNode(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        # Set up cuda and warp
        self.device='cuda'
        self.batch_size = 1
        self.num_actions = 11
        self.num_obs = 159

        # Set the warp cache directory based on device
        warp_cache_dir = ""
        initialize_warp(self.device)

        # For converting ROS image messages to CV formates
        self.bridge = CvBridge()

        # Camera subscriber
        self._left_image_lock = Lock()
        self._right_image_lock = Lock()
        self._left_image = None
        self._right_image = None
        self._image_height = 480
        self._image_width = 640

        # NOTE: expecting 1/2 the resolution
        self._downsample_factor = 2
        self.camera_left_feedback_time = time.time()
        self.camera_left_sub = self.create_subscription(
            Image,
            '/robot2/camera2/color/image_raw',
            self._left_camera_callback,
            10
        )

        self.camera_right_feedback_time = time.time()
        self.camera_right_sub = self.create_subscription(
            Image,
            '/robot3/camera3/color/image_raw',
            self._right_camera_callback,
            10
        )

        # Robot subscribers
        self.robot_q = torch.zeros(self.batch_size, 23, device=self.device)
        self.robot_qd = torch.zeros(self.batch_size, 23, device=self.device)
        self.kuka_feedback_time = time.time()
        self.allegro_feedback_time = time.time()
        self._kuka_joint_position_lock = Lock()
        self._allegro_joint_position_lock = Lock()
        self._kuka_sub = self.create_subscription(
            JointState(),
            '/kuka/joint_states',
            self._kuka_sub_callback,
            1)
        self._allegro_sub = self.create_subscription(
            JointState(),
            '/allegro/joint_states',
            self._allegro_sub_callback,
            1)

        # Fabric subscribers
        self.fabric_q = torch.zeros(self.batch_size, 23, device=self.device)
        self.fabric_qd = torch.zeros(self.batch_size, 23, device=self.device)
        self.fabric_qdd = torch.zeros(self.batch_size, 23, device=self.device)
        self.fabric_feedback_time = time.time()
        self._fabric_feedback_lock = Lock()

        self._fabric_sub = self.create_subscription(
            JointState(),
            '/kuka_allegro_fabric/joint_states',
            self._fabric_sub_callback,
            1)

        self._publish_rate = 60. # Hz
        self._publish_dt = 1./self._publish_rate # s

        # Goal to bring object to. NOTE: this should be a command in the future
        self.object_goal =\
            torch.tensor([-0.5, 0., 0.75], device=self.device).repeat((self.batch_size, 1))

        # For querying 3D points on hand
        hand_body_names = [
            "palm_link",
            "index_biotac_tip",
            "middle_biotac_tip",
            "ring_biotac_tip",
            "thumb_biotac_tip",
        ]

        robot_dir_name = "kuka_allegro"
        robot_name = "kuka_allegro"
        urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(
            urdf_path,
            hand_body_names,
            self.batch_size,
            self.device)

        self._hand_points_lock = Lock()


        max_pose_angle = 45.
        PALM_POSE_MINS = PALM_POSE_MINS_FUNC(max_pose_angle)
        PALM_POSE_MAXS = PALM_POSE_MAXS_FUNC(max_pose_angle)

        self.palm_pose_lower_limits = torch.tensor(
            PALM_POSE_MINS, 
            device=self.device
        )
        self.palm_pose_upper_limits = torch.tensor(
            PALM_POSE_MAXS,
            device=self.device
        )

        self.hand_pca_lower_limits = torch.tensor(
            HAND_PCA_MINS,
            device=self.device
        )
        self.hand_pca_upper_limits = torch.tensor(
            HAND_PCA_MAXS,
            device=self.device
        )
        
        self.last_actions = torch.zeros(self.batch_size, 11, device=self.device)

        # FGP publishers
        # For commanding pose target
        self.palm_pose_targets = None
        self._fgp_pose_lock = Lock()
        self._fgp_pose_command_pub = self.create_publisher(
            JointState,
            '/kuka_allegro_fabric/pose_commands',
            1)
        self._fgp_pose_timer = self.create_timer(self._publish_dt, self._fgp_pose_pub_callback)

        # Subscriber for getting PCA commands
        self.hand_pca_targets = None
        self._fgp_pca_lock = Lock()
        self._fgp_pca_command_pub = self.create_publisher(
            JointState,
            '/kuka_allegro_fabric/pca_commands',
            1)
        self._fgp_pca_timer = self.create_timer(self._publish_dt, self._fgp_pca_pub_callback)

        # Create publisher for predicted object pose
        self._object_pos_lock = Lock()
        self.object_pos = None
#        self._object_pos_pub = self.create_publisher(
#            PoseStamped,
#            "/kuka_allegro_fabric/predicted_obj_pose",
#            1)
        self.object_pos_tf = TransformBroadcaster(self)
        self._fgp_object_pos_timer = self.create_timer(self._publish_dt,
                                                       self._object_pos_callback)

        # Create subscriber for activating, deactivating FGP
        self._engage_fgp_lock = Lock()
        self.engage_fgp = False
        self._engage_fgp_sub = self.create_subscription(
            Bool,
            '/engage_fgp',
            self._engage_fgp_callback,
            1)

        # Instantiate FGP
        self.init_fgp()

    def init_fgp(self):
        # get path to config file
        #parent_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
        parent_path = str(pathlib.Path(__file__).parent.resolve())
        parent_path = parent_path.replace("deployment_scripts", "")
        agent_cfg_folder = "tasks/dextrah_kuka_allegro/agents"
        student_cfg_path = os.path.join(
            parent_path,
            agent_cfg_folder,
            #"rl_games_ppo_lstm_scratch_cnn_aux_stereo.yaml",
            "rl_games_ppo_stereo_transformer.yaml"
        )

        # get path to checkpoint
        # NOTE: This assumes that in the root directory of dextrah_lab, the checkpoint is stored in a folder called pretrained_ckpts
        # First RGB student that transferred
        #student_ckpt = "pretrained_ckpts/dextrah_student_85000_iters.pth"
        #student_ckpt = "pretrained_ckpts/dextrah_student_nov1_3_iters.pth"
        #student_ckpt = "pretrained_ckpts/dextrah_student_66000_iters.pth"
        #student_ckpt = "pretrained_ckpts/rnn/new_dextrah_student_4.pth"
        #student_ckpt = "pretrained_ckpts/rnn/student_2_flipped.pth"
        #student_ckpt = "pretrained_ckpts/rnn/dextrah_10x_seed3.pth"
        #student_ckpt = "pretrained_ckpts/rnn/dextrah_512_kpt_seed_3.pth"
        
        #student_ckpt = "pretrained_ckpts/rnn/dextrah_new_arch_scratch_cnn_seed_3.pth"
        
        # NOTE: current best
        #student_ckpt = "pretrained_ckpts/rnn/dextrah_new_arch_resnet18_seed_3.pth"

        #student_ckpt = "pretrained_ckpts/rnn/wider_large_scale_pos_embed_seed_4.pth"
        #student_ckpt = "pretrained_ckpts/rnn/wider_large_scale_seed_4.pth"
        
        #student_ckpt = "pretrained_ckpts/rnn/new_obj_prims_seed_7.pth"
        #student_ckpt = "pretrained_ckpts/rnn/stereo_aux_10_seed_7.pth"
        student_ckpt = "pretrained_ckpts/rnn/student_1_TEST.pth"
        
        #student_ckpt = "pretrained_ckpts/rnn/dextrah_resnet_scaled_seed_7.pth"
        
        #student_ckpt = "pretrained_ckpts/rnn/dextrah_resnet_kpt_seed_2.pth"
        
        #student_ckpt = "pretrained_ckpts/rnn/dexrah_resnet_scale_seed_4.pth"

        #student_ckpt = "pretrained_ckpts/rnn/dextrah_8192_kpt_seed_2.pth"
        #student_ckpt = "pretrained_ckpts/rnn/new_dextrah_student_60000_iters.pth"
        #student_ckpt = "pretrained_ckpts/rnn/dextrah_student_109000_iters.pth"
        #student_ckpt = "pretrained_ckpts/rnn/new_dextrah_student_18000_iters.pth"
        #student_ckpt = "pretrained_ckpts/rnn/new_dextrah_student_cam_jitter.pth"
        #student_ckpt = "pretrained_ckpts/rnn/new_dextrah_student_2k_1.pth"
        #student_ckpt = "pretrained_ckpts/rnn/dextrah_student_4000_iters.pth"
        #student_ckpt = "pretrained_ckpts/rnn/dextrah_student_152000_iters.pth"
        #student_ckpt = "pretrained_ckpts/dextrah_student_24000_iters.pth"
        #student_ckpt = "pretrained_ckpts/dextrah_student_16000_iters.pth"
        student_ckpt_path = os.path.join(
            parent_path,
            student_ckpt
        )

        # register our custom model with the rl_games model builder
        #model_builder.register_network("a2c_aux_cnn_net_stereo", A2CWithAuxCNNStereoBuilder)
        model_builder.register_network("a2c_stereo_transformer", A2CStereoTransformerBuilder)

        num_proprio_obs = 159
        num_actions = 11
        img_shape = (3, 240, 320)
        # create the model
        #self.dextrah_fgp = DextrahFGP(
        self.dextrah_fgp = DextrAHFGP(
            cfg_path=student_cfg_path,
            img_shape=img_shape,
            num_proprio_obs=self.num_obs,
            num_actions=self.num_actions,
            ckpt_path=student_ckpt_path,
            device=self.device
        )

        # Reset hidden state
        self.dextrah_fgp.reset_hidden_state()

        # Perform cuda graph capture of FGP
        self.dextrah_fgp.setup_cuda_graph()

    def _left_camera_callback(self, msg):
        '''
        TODO: UPDATE
        '''
        # Convert ROS camera data to np array of shape height, width, channel
        #img_np = np.frombuffer(
        #    msg.data, dtype=np.uint8).reshape((self._image_height, self._image_width, 3)).astype(np.float32)

        # Convert ROS image to numpy image with h x w x 3, and rgb ordering
        img_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8').astype(np.float32)

        # Interpolate down to the target image size
        img_np = cv2.resize(
            img_np,
            (self._image_width//self._downsample_factor, self._image_height//self._downsample_factor),
            interpolation=cv2.INTER_LINEAR
        )

#        bgr_image = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
#        cv2.imshow('Left RGB Image', bgr_image)
#
#        cv2.waitKey(1)

        # Reshape into (3, height, width)
        img_np = np.transpose(img_np, (2, 0, 1))

        # Scale to be between [0, 1]
        img_np /= 255.

        # Move to torch tensor and make sure shape is 1x3xhxw
        with self._left_image_lock:
            self.camera_left_feedback_time = time.time()
            self._left_image = torch.from_numpy(img_np).to(self.device).unsqueeze(0)

    def _right_camera_callback(self, msg):
        '''
        TODO: UPDATE
        '''
        # Convert ROS camera data to np array of shape height, width, channel
        #img_np = np.frombuffer(
        #    msg.data, dtype=np.uint8).reshape((self._image_height, self._image_width, 3)).astype(np.float32)

        # Convert ROS image to numpy image with h x w x 3, and rgb ordering
        img_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8').astype(np.float32)

        # Interpolate down to the target image size
        img_np = cv2.resize(
            img_np,
            (self._image_width//self._downsample_factor, self._image_height//self._downsample_factor),
            interpolation=cv2.INTER_LINEAR
        )

#        bgr_image = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
#        cv2.imshow('Right RGB Image', bgr_image)
#
#        cv2.waitKey(1)

        # Reshape into (3, height, width)
        img_np = np.transpose(img_np, (2, 0, 1))

        # Scale to be between [0, 1]
        img_np /= 255.

        # Move to torch tensor and make sure shape is 1x3xhxw
        with self._right_image_lock:
            self.camera_right_feedback_time = time.time()
            self._right_image = torch.from_numpy(img_np).to(self.device).unsqueeze(0)
            # Now flip
            self._right_image = torch.flip(self._right_image, dims=(2,3))

    def _kuka_sub_callback(self, msg):
        """
        Acquires the feedback time, sets the measured joint position for the
        kuka, and also sets the command for the kuka to this measured position
        if a command does not yet exist.
        ------------------------------------------
        :param msg: ROS 2 JointState message type
        """

        # Create GPU pytorch tensors
        position_tensor = torch.tensor(
            np.array([msg.position]), device=self.device
            )
        velocity_tensor = torch.tensor(
            np.array([msg.velocity]), device=self.device
            )

        # Copy over to robot state tensors
        with self._kuka_joint_position_lock:
            self.kuka_feedback_time = time.time()
            self.robot_q[:, :7].copy_(position_tensor)
            self.robot_qd[:, :7].copy_(velocity_tensor)

    def _allegro_sub_callback(self, msg):
        """
        Acquires the feedback time, sets the measured joint position for the
        allegro, and also sets the command for the allegro to this measured position
        if a command does not yet exist.
        ------------------------------------------
        :param msg: ROS 2 JointState message type
        """
        # Create GPU pytorch tensors
        position_tensor = torch.tensor(
            np.array([msg.position]), device=self.device
            )
        velocity_tensor = torch.tensor(
            np.array([msg.velocity]), device=self.device
            )

        # Copy over to robot state tensors
        with self._allegro_joint_position_lock:
            self.allegro_feedback_time = time.time()
            self.robot_q[:, 7:].copy_(position_tensor)
            self.robot_qd[:, 7:].copy_(velocity_tensor)


    def _fabric_sub_callback(self, msg):
        """
        Acquires the fabric state."
        """
        # Create GPU pytorch tensors
        position_tensor = torch.tensor(
            np.array([msg.position]), device=self.device
            )
        velocity_tensor = torch.tensor(
            np.array([msg.velocity]), device=self.device
            )
        acceleration_tensor = torch.tensor(
            np.array([msg.effort]), device=self.device
            )

        # Copy over to fabric state tensors
        with self._fabric_feedback_lock:
            self.fabric_feedback_time = time.time()
            self.fabric_q.copy_(position_tensor)
            self.fabric_qd.copy_(velocity_tensor)
            self.fabric_qdd.copy_(acceleration_tensor)

    def _fgp_pose_pub_callback(self):
        """
        Publishes latest FGP pose command.
        """
        palm_pose_targets = None
        with self._fgp_pose_lock:
            if self.palm_pose_targets is not None:
                palm_pose_targets = self.palm_pose_targets[0,:].float().detach().cpu().numpy()

        if palm_pose_targets is not None:
            engage_fgp = False
            with self._engage_fgp_lock:
                engage_fgp = self.engage_fgp

            if engage_fgp:
                #print('palm-----------------', list(palm_pose_targets))
                msg = JointState()
                msg.name = ['x', 'y', 'z', 'xrot', 'yrot', 'zrot']
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.position = palm_pose_targets.tolist()
                msg.velocity = []
                msg.effort = []
                self._fgp_pose_command_pub.publish(msg)

    def _fgp_pca_pub_callback(self):
        """
        Publishes latest FGP pca command.
        """
        hand_pca_targets = None
        with self._fgp_pca_lock:
            if self.hand_pca_targets is not None:
                hand_pca_targets = self.hand_pca_targets[0,:].float().detach().cpu().numpy()
        

        if hand_pca_targets is not None:
            engage_fgp = False
            with self._engage_fgp_lock:
                engage_fgp = self.engage_fgp

            if engage_fgp:
                msg = JointState()
                msg.name = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5']
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.position = hand_pca_targets.tolist()
                msg.velocity = []
                msg.effort = []
                self._fgp_pca_command_pub.publish(msg)

    def _object_pos_callback(self):
        """
        Publishes the latest predicted object position
        """
        object_pos = None
        with self._object_pos_lock:
            if self.object_pos is not None:
                object_pos = self.object_pos[0,:].float().detach().cpu().numpy()

        if object_pos is not None:
            engage_fgp = False
            with self._engage_fgp_lock:
                engage_fgp = self.engage_fgp

            if engage_fgp:
    #            pose_msg = PoseStamped()
    #            pose_msg.header.stamp = self.get_clock().now().to_msg()
    #            pose_msg.header.frame_id = 'robot_base'# Set the frame ID
    #
    #            pose_msg.pose.position.x = float(object_pos[0])
    #            pose_msg.pose.position.y = float(object_pos[1])
    #            pose_msg.pose.position.z = float(object_pos[2])
    #            pose_msg.pose.orientation.x = 0.0
    #            pose_msg.pose.orientation.y = 0.0
    #            pose_msg.pose.orientation.z = 0.0
    #            pose_msg.pose.orientation.w = 1.0
    #
    #            # Publish the message
    #            self._object_pos_pub.publish(pose_msg)
                # Create the transform message
                t = TransformStamped()

                # Set the header information
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'robot_base'
                t.child_frame_id = 'obj_pos'

                # Set the translation (x, y, z)
                t.transform.translation.x = float(object_pos[0])
                t.transform.translation.y = float(object_pos[1])
                t.transform.translation.z = float(object_pos[2])

                # Set the rotation as a quaternion (x, y, z, w)
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0

                # Broadcast the transform
                self.object_pos_tf.sendTransform(t)

    def _engage_fgp_callback(self, msg):
        with self._engage_fgp_lock:
            self.engage_fgp = msg.data

    def _compute_hand_points(self):
        # Noisy hand point position and velocity as calculated from fabric taskmap
        robot_q = None
        robot_qd = None
        feedback_timed_out = False
        with self._kuka_joint_position_lock:
            with self._allegro_joint_position_lock:
                robot_q = self.robot_q.clone()
                robot_qd = self.robot_qd.clone()

                end = time.time()
                if (end - self.kuka_feedback_time) > (3. * self._publish_dt):
                    print('no feedback from kuka')
                    feedback_timed_out = True
                
                if (end - self.allegro_feedback_time) > (3. * self._publish_dt):
                    print('no feedback from allegro')
                    feedback_timed_out = True

        hand_pos, hand_points_jac = self.hand_points_taskmap(
            robot_q, None)
        hand_vel = torch.bmm(
            hand_points_jac, robot_qd.unsqueeze(2)).squeeze(2)

        return hand_pos, hand_vel, robot_q, robot_qd, feedback_timed_out

    def compute_fgp_observation(self):

        feedback_timed_out = False

        # Compute hand point data
        hand_pos, hand_vel, robot_q, robot_qd, feedback_timed_out = self._compute_hand_points()

        fabric_q = None
        fabric_qd = None
        fabric_qdd = None

        with self._fabric_feedback_lock:
            
            # Make sure we are getting all the feedback signals
            end = time.time()
            
            if (end - self.fabric_feedback_time) > (3. * self._publish_dt):
                print('no feedback from fabric')
                feedback_timed_out = True

            fabric_q = self.fabric_q.clone()
            fabric_qd = self.fabric_qd.clone()
            fabric_qdd = self.fabric_qdd.clone()

        state = torch.cat(
            (
                robot_q,
                robot_qd * 0.,
                hand_pos, # 46:61
                hand_vel * 0., # 61:76
                # object goal
                self.object_goal, # 76:79
                # last action
                self.last_actions, # 79:90
                # fabric states
                fabric_q, # 90:113
                fabric_qd * 0., # 113:136
                fabric_qdd * 0.# 136:159
            ),
            dim=-1
        )

        # TODO: undo this
        #state *= 0.

        # Copy camera image
        left_image = None
        with self._left_image_lock:
            end = time.time()
            if (end - self.camera_left_feedback_time) > (3. * self._publish_dt): 
                print('no feedback from left camera')
                feedback_timed_out = True
            left_image = torch.clone(self._left_image)

        right_image = None
        with self._right_image_lock:
            end = time.time()
            if (end - self.camera_right_feedback_time) > (3. * self._publish_dt): 
                print('no feedback from right camera')
                feedback_timed_out = True
            right_image = torch.clone(self._right_image)

        return state, left_image, right_image, feedback_timed_out

    def compute_actions(self, state, left_image, right_image, transmit=True, save_pth=False):
#        with torch.no_grad():
#            action_dict = self.dextrah_fgp.step(state, left_image, right_image)

        action_dict = self.dextrah_fgp.step_cuda_graph(state, left_image, right_image)


        # NOTE: pulling out mean action. could use the selected_action
        # instead if you want a stochastic policy
        actions = action_dict["mus"]
        #actions = action_dict["selected_action"]

        # Now clip the actions between -1 and 1
        actions = torch.clamp(actions, min=-1, max=1)

        has_nan = torch.isnan(actions).any()
        has_inf = torch.isinf(actions).any()
        
        if has_nan:
            print('NaNing!!!')
        if has_inf:
            print('Infing!!!')

        object_pos = action_dict["obj_pos"]

        #print(object_pos)

        assert_equals(actions.shape, (self.batch_size, 11))

        # Slice out the actions for the palm and the hand
        palm_actions = actions[:, : (NUM_XYZ + NUM_RPY)]
        hand_actions = actions[
            :, (NUM_XYZ + NUM_RPY) : (NUM_HAND_PCA + NUM_XYZ + NUM_RPY)
        ]

        # Update the action target tensors
        if transmit:
            with self._fgp_pose_lock:
                if self.palm_pose_targets is None:
                    self.palm_pose_targets = compute_absolute_action( 
                        raw_actions=palm_actions,
                        lower_limits=self.palm_pose_lower_limits,
                        upper_limits=self.palm_pose_upper_limits,
                        )
                else:
                    # In-place update to palm pose targets
                    self.palm_pose_targets.copy_(
                        compute_absolute_action(
                            raw_actions=palm_actions,
                            lower_limits=self.palm_pose_lower_limits,
                            upper_limits=self.palm_pose_upper_limits,
                        )
                    )

            with self._fgp_pca_lock: 
                if self.hand_pca_targets is None:
                    self.hand_pca_targets = compute_absolute_action(
                        raw_actions=hand_actions,
                        lower_limits=self.hand_pca_lower_limits,
                        upper_limits=self.hand_pca_upper_limits,
                    )
                else:
                    # In-place update to hand PCA targets
                    self.hand_pca_targets.copy_(
                        compute_absolute_action(
                            raw_actions=hand_actions,
                            lower_limits=self.hand_pca_lower_limits,
                            upper_limits=self.hand_pca_upper_limits,
                        )
                    )

            with self._object_pos_lock:
                if self.object_pos is None:
                    self.object_pos = object_pos.clone()
                else:
                    self.object_pos.copy_(object_pos)

            # Update last action
            self.last_actions = actions.clone()

    def burn_in(self):
        feedback_timed_out = True
        for i in range(5):
            # Pack and prepare observations
            state, left_image, right_image, feedback_timed_out = self.compute_fgp_observation()

            # Query the FGP for actions
            # NOTE: publisher callbacks will pull action data from tensors
            # into lists themselves and publish
            if not feedback_timed_out:
                print('copmute actions')
                self.compute_actions(state, left_image, right_image)
            else:
                print('not computingn actions')
                self.compute_actions(state, left_image, right_image, transmit=False)

            time.sleep(1./60)

        if feedback_timed_out is True:
            print('Failed to burn in policy due to lack of obs')
            sys.exit()

        # Clear memory
        self.dextrah_fgp.reset_hidden_state()

    def run(self):
        # Main control loop
        control_iter = 0
        print_iter = 60
        loop_time_filtered = 0.

        # Burn in
        print('Burning in')
        self.burn_in()

        print('Engaging policy')
        while rclpy.ok():

            # Set time start of loop
            start = time.time()

            # Pack and prepare observations
            state, left_image, right_image, feedback_timed_out = self.compute_fgp_observation()

            #img_dict = {'left_img': left_image,
            #            'right_img': right_image}
            #torch.save(img_dict, 'images.pth')

#            break

            # Query the FGP for actions
            # NOTE: publisher callbacks will pull action data from tensors
            # into lists themselves and publish
            # TODO: if times out again, probably should send robot to home position
            if not feedback_timed_out:
                self.compute_actions(state, left_image, right_image)
            else:
                print('not computingn actions')

            # Reset hidden state if not engaging FGP
            with self._engage_fgp_lock:
                if self.engage_fgp is False:
                    #print('resetting hidden state')
                    self.dextrah_fgp.reset_hidden_state()

            # Keep 60 Hz tick rate
            while (time.time() - start) < self._publish_dt:
                time.sleep(.00001)

            # Print control loop frequencies
            loop_time = time.time() - start
            alpha = 0.5
            if control_iter == 0:
                loop_time_filtered = loop_time
            else:
                loop_time_filtered = alpha * loop_time + (1. - alpha) * loop_time_filtered
            if (control_iter % print_iter) == 0:
                print('avg control rate', 1./loop_time_filtered)

            control_iter += 1

    def test_spinning(self):

        while rclpy.ok():
            start = time.time()
            # Keep 60 Hz tick rate
            while (time.time() - start) < self._publish_dt:
                time.sleep(.00001)

if __name__ == "__main__":
    print("Starting DextrAH FGP node")
    rclpy.init()

    # Create the fabric
    node_name = "dextrah_fgp_stereo"
    dextrah_fgp_node = DextrahFGPNode(node_name)

    # Spawn separate thread that spools the fabric
    spin_thread = Thread(target=rclpy.spin, args=(dextrah_fgp_node,), daemon=True)
    spin_thread.start()
    
    # Give time for data to flow
    time.sleep(1.)

    # Start the main dextrah loop
    dextrah_fgp_node.run()
    #dextrah_fgp_node.test_spinning()

    # Destroy node and shut down ROS
    dextrah_fgp_node.destroy_node()
    rclpy.shutdown()

    print('DextrAH FGP closed.')

