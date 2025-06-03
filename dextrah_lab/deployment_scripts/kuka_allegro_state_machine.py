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
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

# Third party
import torch
import yaml
import time
import numpy as np

# Fabrics
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.utils.utils import initialize_warp

class DextrahStateMachineNode(Node):
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

        # Rates
        self._publish_rate = 60. # Hz
        self._publish_dt = 1./self._publish_rate # s

        # For querying 3D point on hand
        hand_body_names = [
            "palm_link",
            "index_biotac_tip",
            "thumb_biotac_tip"
        ]

        robot_dir_name = "kuka_allegro"
        robot_name = "kuka_allegro"
        urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(
            urdf_path,
            hand_body_names,
            self.batch_size,
            self.device)

        # Object position predictions
        self._obj_pos_lock = Lock()
        self.obj_feedback_time = time.time()
        self.obj_pos = None
        self.obj_pos_sub = self.create_subscription(
            TFMessage(),
            '/tf',  # Replace with your TF topic
            self._tf_callback,
            10
        )

        # Fabric subscribers
        self._fabric_feedback_lock = Lock()
        self.fabric_q = torch.zeros(self.batch_size, 23, device=self.device)
        #self.fabric_qd = torch.zeros(self.batch_size, 23, device=self.device)
        #self.fabric_qdd = torch.zeros(self.batch_size, 23, device=self.device)
        self.fabric_feedback_time = time.time()

        self._fabric_sub = self.create_subscription(
            JointState(),
            '/kuka_allegro_fabric/joint_states',
            self._fabric_sub_callback,
            1)

        # Subscriber for getting PCA commands
        self._pca_listener_lock = Lock()
        self._kuka_allegro_pca_command_sub = self.create_subscription(
            JointState(),
            '/kuka_allegro_fabric/pca_commands',
            self._kuka_allegro_fabric_pca_command_sub_callback,
            1)

        # Action publishers
        self._engage_fgp_lock = Lock()
        self.engage_fgp = False
        self.engage_fgp_pub = self.create_publisher(Bool, '/engage_fgp', 1)
        self._engage_fgp_timer = self.create_timer(self._publish_dt, self._engage_fgp_callback)

        # For commanding pose target
        self.palm_pose_targets = None
        self._pose_lock = Lock()
        self._pose_command_pub = self.create_publisher(
            JointState,
            '/kuka_allegro_fabric/pose_commands',
            1)
        self._pose_timer = self.create_timer(self._publish_dt, self._pose_pub_callback)

        # Subscriber for getting PCA commands
        self.hand_pca_targets = None
        self._pca_lock = Lock()
        self._pca_command_pub = self.create_publisher(
            JointState,
            '/kuka_allegro_fabric/pca_commands',
            1)
        self._pca_timer = self.create_timer(self._publish_dt, self._pca_pub_callback)

        # State machine states
        # Set a default pose target to revert to after successfully grasping object
        self.default_pose_target = \
            np.array([-0.6868,  0.0320,  0.685, -2.3873, -0.0824,  3.1301]).astype(float)
        
        self.default_pca_target = \
            np.array([1.5, 1.5, 0., 0.5, -0.25]).astype(float)
        
        self._pca_filtering_lock = Lock()
        self.pca_target_filtered = \
            np.array([1.5, 1.5, 0., 0.5, -0.25]).astype(float)

        self.box_pose_target = \
            np.array([0.02,  0.45,  0.385, 3.14*(3./4), 0.,  3.14]).astype(float)
        
        self.open_pca_target = \
            np.array([0.2475, -0.3286, -0.7238, -0.0192, -0.5532]).astype(float)

        self.terminal_pca_target = None
        self.move_counter = 0
        self.moving_to_box = False
        self.moving_to_home = True
        self.opening_hand = False

    def _tf_callback(self, msg):
        # Get the XYZ location of the object prediction
        if len(msg.transforms) > 0:
            for tf in msg.transforms:
                if tf.child_frame_id == 'obj_pos':
                    with self._obj_pos_lock:
                        # Pull out translation
                        self.obj_pos = np.array([tf.transform.translation.x,
                                                 tf.transform.translation.y,
                                                 tf.transform.translation.z])

                        self.obj_feedback_time = time.time()

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
            #self.fabric_qd.copy_(velocity_tensor)
            #self.fabric_qdd.copy_(acceleration_tensor)

    def _kuka_allegro_fabric_pca_command_sub_callback(self, msg):
        """
        Sets the PCA position target coming in from the ROS topic.
        ------------------------------------------
        :param msg: ROS 2 JointState message type
        """
        new_pca_action = None
        with self._pca_listener_lock:
            #self.hand_target.copy_(torch.tensor([list(msg.position)], device=self.device))
            new_pca_action = np.array(msg.position)

        with self._pca_filtering_lock:
            # Filter PCA target
            alpha = .1
            self.pca_target_filtered = alpha * new_pca_action +\
                                       (1. - alpha) * self.pca_target_filtered

    def _engage_fgp_callback(self):
        msg = Bool()
        with self._engage_fgp_lock:
            msg.data = self.engage_fgp
        self.engage_fgp_pub.publish(msg)

    def _pose_pub_callback(self):
        """
        Publishes latest pose command.
        """
        palm_pose_targets = None
        with self._pose_lock:
            if self.palm_pose_targets is not None:
                palm_pose_targets = copy.deepcopy(self.palm_pose_targets)

        if palm_pose_targets is not None:
            engage_fgp = False
            with self._engage_fgp_lock:
                engage_fgp = self.engage_fgp

            # If FGP not being engaged, the transmit pose targets set in this script
            if not engage_fgp:
                #print('palm-----------------', list(palm_pose_targets))
                msg = JointState()
                msg.name = ['x', 'y', 'z', 'xrot', 'yrot', 'zrot']
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.position = palm_pose_targets.tolist()
                msg.velocity = []
                msg.effort = []
                self._pose_command_pub.publish(msg)

    def _pca_pub_callback(self):
        """
        Publishes latest pca command.
        """
        hand_pca_targets = None
        with self._pca_lock:
            if self.hand_pca_targets is not None:
                hand_pca_targets = copy.deepcopy(self.hand_pca_targets)
        
        if hand_pca_targets is not None:
            engage_fgp = False
            with self._engage_fgp_lock:
                engage_fgp = self.engage_fgp

            if not engage_fgp:
                msg = JointState()
                msg.name = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5']
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.position = hand_pca_targets.tolist()
                msg.velocity = []
                msg.effort = []
                self._pca_command_pub.publish(msg)

    def compute_hand_pos(self):
        joint_pos = None
        with self._fabric_feedback_lock:
            joint_pos = torch.clone(self.fabric_q)

        hand_pos, _ = self.hand_points_taskmap(joint_pos, None)

        hand_pos_np = hand_pos[0].detach().cpu().numpy()

        palm_pos = hand_pos_np[:3]
        index_pos = hand_pos_np[3:6]
        thumb_pos = hand_pos_np[6:9]

        return (palm_pos, index_pos, thumb_pos)

    def decide_action(self, obj_pos, palm_pos, index_pos, thumb_pos):


#        if self.move_counter % 180 == 0:
#            with self._engage_fgp_lock:
#                self.engage_fgp = not self.engage_fgp

        # Moving to drop off into box
        if self.moving_to_box:
            print('Dropping off part')
            #self.move_counter += 1

            # Transmit box pose target and filtered PCA target
            with self._pose_lock:
                self.palm_pose_targets = copy.deepcopy(self.box_pose_target)
            with self._pca_lock:
                if self.opening_hand == True:
                    self.hand_pca_targets = copy.deepcopy(self.open_pca_target)
                else:
                    #self.hand_pca_targets = copy.deepcopy(self.default_pca_target)
                    self.hand_pca_targets = copy.deepcopy(self.terminal_pca_target)
        
            # Calculate how far palm point is from target
            palm_pos_error = np.linalg.norm(self.box_pose_target[:3] - palm_pos)
            print(palm_pos_error)

            # Calculate distance between index and thumb as measure of opening aperture
            finger_aperture = np.linalg.norm(index_pos - thumb_pos)

            if palm_pos_error < .2 and not self.opening_hand:
                self.opening_hand = True
            elif palm_pos_error < .2 and self.opening_hand and finger_aperture > .23: 
                self.moving_to_home = True
                self.moving_to_box = False
                self.opening_hand = False

        # Moving to home
        elif self.moving_to_home:
            print('Going home')
            #self.move_counter += 1

            # Copy over pose box target
            # Moving to home
            with self._pose_lock:
                self.palm_pose_targets = copy.deepcopy(self.default_pose_target)
            with self._pca_lock:
                self.hand_pca_targets = copy.deepcopy(self.default_pca_target)

            palm_pos_error = np.linalg.norm(self.default_pose_target[:3] - palm_pos)
            
            if palm_pos_error < .1:
                self.moving_to_home = False
                self.moving_to_box = False
                with self._engage_fgp_lock:
                    self.engage_fgp = True

        else:
            # Deploying policy
            print('Following RL')
            
            # TODO: complete a timeout to return home
#            self.move_counter += 1
#
#            if self.move_counter > int(20 * 60):


            # If predicted object position is above some height, then transition
            # to moving to box
            # NOTE: this will still execute the RL actions this pass

#            self.moving_to_box = True
#
#            # Deactivate the FGP
#            with self._engage_fgp_lock:
#                self.engage_fgp = False

            if obj_pos is not None:
                if obj_pos[2] > 0.45:
                    self.moving_to_box = True

                    # Set PCA actions to this set
                    #self.terminal_pca_target = torch.clone(palm_pose_hand_pca_targets[6:])
                    with self._pca_filtering_lock:
                        self.terminal_pca_target = copy.deepcopy(self.pca_target_filtered)

                    # Deactivate the FGP
                    with self._engage_fgp_lock:
                        self.engage_fgp = False

#            elif palm_pos[0,0] >= -0.3: # RL policy is quitting
#                # Copy over pose box target
#                palm_pose_hand_pca_targets[:6] = torch.clone(self.default_pose_target)
#                palm_pose_hand_pca_targets[6:] = torch.clone(self.default_pca_target)
#
#                # Turning off moving to box and resetting counter
#                self.moving_to_box = False
#                self.move_counter = 0
#
#                # Turn on moving to home
#                self.moving_to_home = True
#            else:
#                # Pass through and we execute RL actions directly
#                pass

    def run(self):
        # Main control loop
        control_iter = 0
        print_iter = 60
        loop_time_filtered = 0.

        print('Engaging state machine')
        while rclpy.ok():
            # Set time start of loop
            start = time.time()

            # Calculate feedback signals for decision making
            # Get object position
            obj_pos = None
            with self._obj_pos_lock:
                if (time.time() - self.obj_feedback_time) > 1.:
                    obj_pos = None
                else:
                    obj_pos = copy.deepcopy(self.obj_pos)

            # Get palm position
            palm_pos, index_pos, thumb_pos = self.compute_hand_pos()

            # Decide actions
            self.decide_action(obj_pos, palm_pos, index_pos, thumb_pos)

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

if __name__ == "__main__":
    print("Starting DextrAH state machine node")
    rclpy.init()

    # Create the fabric
    node_name = "dextrah_state_machine"
    dextrah_state_machine_node = DextrahStateMachineNode(node_name)

    # Spawn separate thread that spools the fabric
    spin_thread = Thread(target=rclpy.spin, args=(dextrah_state_machine_node,), daemon=True)
    spin_thread.start()
    
    # Give time for data to flow
    time.sleep(1.)

    # Start the main dextrah loop
    dextrah_state_machine_node.run()

    # Destroy node and shut down ROS
    dextrah_state_machine_node.destroy_node()
    rclpy.shutdown()

    print('DextrAH state machine closed.')

