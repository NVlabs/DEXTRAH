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
    NUM_HAND_PCA
)

# Dextrah FGP
from dextrah_lab.distillation.a2c_with_aux_cnn import A2CBuilder as A2CWithAuxCNNBuilder

def load_param_dict(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class DextrahFGP:
    def __init__(
        self, cfg_path, num_proprio_obs,
        num_actions, ckpt_path, device
    ):
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.device = device

        # read the yaml file
        network_params = load_param_dict(cfg_path)["params"]

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
        }

        # build the model
        builder = ModelBuilder()
        network = builder.load(network_params)
        self.model = network.build(model_config).to(self.device)

        # load checkpoint if available
        if ckpt_path is not None:
            weights = torch_ext.load_checkpoint(ckpt_path)
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

    def step(self, proprio, clamped_depth, camera_type):
        # package observations
        #clamped_depth = torch.clamp(
        #    depth, min=0.5, max=1.5
        #)
        batch_dict = None
        if camera_type == "depth":
            batch_dict = {
                "is_train": False,
                "obs": proprio.repeat(2, 1),
                "img": clamped_depth.repeat(2, 1, 1, 1),
                "prev_actions": self.dummy_prev_actions
            }
        elif camera_type == "rgb":
            batch_dict = {
                "is_train": False,
                "obs": proprio.repeat(2, 1),
                "rgb": clamped_depth.repeat(2, 1, 1, 1),
                "prev_actions": self.dummy_prev_actions,
                "img": None
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
        selected_action = distr.sample()


        return {
            "mus": mus,
            "sigmas": sigmas,
            "obj_pos": position,
            "selected_action": selected_action
        }

    def reset_hidden_state(self):
        for i in range(len(self.hidden_states)):
            self.hidden_states[i] *= 0.

class DextrahFGPNode(Node):
    def __init__(self, node_name: str, camera_type: str) -> None:
        super().__init__(node_name)

        #self.depth_test = torch.load('depth_map_sim.pth').unsqueeze(0).unsqueeze(0)

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
        self._image_lock = Lock()
        self._depth_image = None
        self._min_depth = 0.5 # m
        self._max_depth = 1.3 # m
        self._image_height = 480
        self._image_width = 640

        self.camera_type = camera_type
        if camera_type == "rgb":
            # NOTE: expecting 1/2 the resolution
            self._downsample_factor = 2
            self.depth_feedback_time = time.time()
            self.subscription = self.create_subscription(
                Image,
                '/camera1/color/image_raw',
                self._color_camera_callback,
                10
            )
        else:
            # NOTE: Expecting 1/4 the resolution
            self._downsample_factor = 4
            self.depth_feedback_time = time.time()
            self.subscription = self.create_subscription(
                Image,
                '/camera1/aligned_depth_to_color/image_raw',
                self._camera_callback,
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

        self.palm_pose_lower_limits = torch.tensor(
            [-1, -0.75, 0, -np.pi, -np.pi / 2, -np.pi],
            device=self.device
        )
        self.palm_pose_upper_limits = torch.tensor(
            [0.25, 0.75, 1, np.pi, np.pi / 2, np.pi],
            device=self.device
        )
        self.hand_pca_lower_limits = torch.tensor(
            [0.2475, -0.3286, -0.7238, -0.0192, -0.5532],
            device=self.device
        )
        self.hand_pca_upper_limits = torch.tensor(
            [3.8336, 3.0025, 0.8977, 1.0243, 0.0629],
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

        # Instantiate FGP
        self.init_fgp()

    def init_fgp(self):
        # get path to config file
        parent_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
        parent_path = parent_path.replace("deployment_scripts", "")
        agent_cfg_folder = "tasks/dextrah_kuka_allegro/agents"
        student_cfg_path = os.path.join(
            parent_path,
            agent_cfg_folder,
            "rl_games_ppo_lstm_scratch_cnn_aux.yaml"
        )

        # get path to checkpoint
        # NOTE: This assumes that in the root directory of dextrah_lab, the checkpoint is stored in a folder called pretrained_ckpts
        # First depth student that transferred
        #student_ckpt = "pretrained_ckpts/dextrah_student_24000_iters.pth"
        # First RGB student that transferred
        student_ckpt = "pretrained_ckpts/dextrah_student_105000_iters.pth"
        student_ckpt_path = os.path.join(
            parent_path,
            student_ckpt
        )

        # register our custom model with the rl_games model builder
        model_builder.register_network("a2c_aux_cnn_net", A2CWithAuxCNNBuilder)

        # create the model
        self.dextrah_fgp = DextrahFGP(
            cfg_path=student_cfg_path,
            num_proprio_obs=self.num_obs,
            num_actions=self.num_actions,
            ckpt_path=student_ckpt_path,
            device=self.device
        )

        # Reset hidden state
        self.dextrah_fgp.reset_hidden_state()


    def _camera_callback(self, msg):
        '''
        msg: integer values 0-255 (encoding: '16UC1' or 'mono16')
             expected resolution: 640x480
        The learned model expects real values between 0.5 and 1.4
        and with image shape of (batch_size, 1, height, width)
        '''
        # Convert ROS camera data to np array
        img_np = np.frombuffer(
            msg.data, dtype=np.uint16).reshape(self._image_height, self._image_width).astype(np.float32)
        # Interpolate down to the target image size
        img_np = cv2.resize(
            img_np,
            (self._image_width//self._downsample_factor,
             self._image_height//self._downsample_factor),
            interpolation=cv2.INTER_LINEAR
        )
        img_np *= 1e-3

        # Clamp to be within expected limits
        img_np[img_np > self._max_depth] = 0.
        img_np[img_np < self._min_depth] = 0.

        # Move to torch tensor and make sure shape is 1x1xhxw
        with self._image_lock:
            self.depth_feedback_time = time.time()
            self._depth_image = torch.from_numpy(img_np).to('cuda').unsqueeze(0).unsqueeze(0)
            #self._depth_image = self.depth_test.to('cuda')

    def _color_camera_callback(self, msg):
        '''
        msg: integer values 0-255 (encoding: '16UC1' or 'mono16')
             expected resolution: 640x480
        The learned model expects real values between 0.5 and 1.4
        and with image shape of (batch_size, 1, height, width)
        '''
        # Convert ROS camera data to np array of shape height, width, channel
        img_np = np.frombuffer(
            msg.data, dtype=np.uint8).reshape((self._image_height, self._image_width, 3)).astype(np.float32)

        # Interpolate down to the target image size
        img_np = cv2.resize(
            img_np,
            (self._image_width//self._downsample_factor, self._image_height//self._downsample_factor),
            interpolation=cv2.INTER_LINEAR
        )

        # Reshape into (3, height, width)
        img_np = np.transpose(img_np, (2, 0, 1))

        # Scale to be between [0, 1]
        img_np /= 255.

        # Move to torch tensor and make sure shape is 1x1xhxw
        with self._image_lock:
            self.depth_feedback_time = time.time()
            self._depth_image = torch.from_numpy(img_np).to('cuda').unsqueeze(0)

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
                object_pos = self.object_pos[0,:].detach().cpu().numpy()

        if object_pos is not None:
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

    def _compute_hand_points(self):
        # Noisy hand point position and velocity as calculated from fabric taskmap
        robot_q = None
        robot_qd = None
        with self._kuka_joint_position_lock:
            with self._allegro_joint_position_lock:
                robot_q = self.robot_q.clone()
                robot_qd = self.robot_qd.clone()

        hand_pos, hand_points_jac = self.hand_points_taskmap(
            robot_q, None)
        hand_vel = torch.bmm(
            hand_points_jac, robot_qd.unsqueeze(2)).squeeze(2)

        return hand_pos, hand_vel

    def compute_fgp_observation(self):

        feedback_timed_out = False

        # Compute hand point data
        hand_pos, hand_vel = self._compute_hand_points()

        state = None

        with self._kuka_joint_position_lock and\
             self._allegro_joint_position_lock and\
             self._fabric_feedback_lock and\
             self._image_lock:
            
            # Make sure we are getting all the feedback signals
            end = time.time()

            if (end - self.kuka_feedback_time) > (3. * self._publish_dt):
                print('no feedback from kuka')
                feedback_timed_out = True
            
            if (end - self.allegro_feedback_time) > (3. * self._publish_dt):
                print('no feedback from allegro')
                feedback_timed_out = True

            if (end - self.fabric_feedback_time) > (3. * self._publish_dt):
                print('no feedback from fabric')
                feedback_timed_out = True

            if (end - self.depth_feedback_time) > (3. * self._publish_dt): 
                print('no feedback from camera')
                feedback_timed_out = True

            state = torch.cat(
                (
                    self.robot_q,
                    self.robot_qd,
                    hand_pos, # 46:61
                    hand_vel, # 61:76
                    # object goal
                    self.object_goal, # 76:79
                    # last action
                    self.last_actions, # 79:90
                    # fabric states
                    self.fabric_q, # 90:113
                    self.fabric_qd, # 113:136
                    self.fabric_qdd # 136:159
                ),
                dim=-1
            )

        # Copy camera image
        depth_image = None
        with self._image_lock:
            depth_image = torch.clone(self._depth_image)

        return state, depth_image, feedback_timed_out

    def compute_actions(self, state, depth_image, transmit=True, save_pth=False):
#        if save_pth is True:
#            self.dextrah_fgp.reset_hidden_state()
#
#            action_dict = self.dextrah_fgp.step(state, depth_image)
#        
#            actions = action_dict["mus"]
#
#            object_pos = action_dict["obj_pos"]
#
#            dict_to_save = {
#                'state': state,
#                'depth': depth_image,
#                'mu action': actions,
#                'object_pos': object_pos
#            }
#
#            print('Saving input-output dict')
#            torch.save(dict_to_save, 'obs_action_dict.pth')
#
#            sys.exit()

        action_dict = self.dextrah_fgp.step(state, depth_image, self.camera_type)

        # NOTE: pulling out mean action. could use the selected_action
        # instead if you want a stochastic policy
        actions = action_dict["mus"]

        object_pos = action_dict["obj_pos"]



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
            state, depth_image, feedback_timed_out = self.compute_fgp_observation()

            # Query the FGP for actions
            # NOTE: publisher callbacks will pull action data from tensors
            # into lists themselves and publish
            if not feedback_timed_out:
                print('copmute actions')
                self.compute_actions(state, depth_image)
            else:
                print('not computingn actions')
                self.compute_actions(state, depth_image, transmit=False)

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
            state, depth_image, feedback_timed_out = self.compute_fgp_observation()

            # Query the FGP for actions
            # NOTE: publisher callbacks will pull action data from tensors
            # into lists themselves and publish
            # TODO: if times out again, probably should send robot to home position
            if not feedback_timed_out:
                self.compute_actions(state, depth_image) #, save_pth=True)
            else:
                print('not computingn actions')

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
    parser = argparse.ArgumentParser(description="Deploys FGP with depth or color camera.")
    parser.add_argument("--camera", required=True, choices=["rgb", "depth"],
                        help="Specify which kind of camera image, rgb or depth.")
    args = parser.parse_args()

    print("Starting DextrAH FGP node")
    rclpy.init()

    # Create the fabric
    node_name = "dextrah_fgp"
    camera_type = args.camera
    dextrah_fgp_node = DextrahFGPNode(node_name, camera_type=camera_type)

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

