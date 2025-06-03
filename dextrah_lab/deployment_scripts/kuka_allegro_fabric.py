#!/usr/bin/env python
#
# Copyright (c) 2024, Nvidia.  All rights reserved.

from threading import Lock, Thread
import math
import time
import argparse
import sys

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool

# Numpy and torch imports
import numpy as np
import torch

# Fabrics imports
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from fabrics_sim.utils.utils import initialize_warp, capture_fabric

class KukaAllegroFabricNode(Node):
    def __init__(self, speed_mode):
        """
        Creates a kuka allegro fabric for controlling the robot, and hooks to
        the correct ROS 2 topics with publishers and subscribers.
        """
        # Initialize ROS 2 node
        print("initializing KukaAllegroFabricNode ROS2 node")
        start_time = time.time()
        super().__init__('kuka_allegro_fabric')
        elapse_time = time.time() - start_time
        print(f"<done> elapse: {elapse_time}")

        self.iters_per_cycle = 1 # default to "normal" speed
        if speed_mode == "normal":
            self.iters_per_cycle = 1 
        elif speed_mode == "fast":
            self.iters_per_cycle = 2

        # Set the GPU device
        self.device = 'cuda'

        # Set the warp cache directory based on device
        warp_cache_dir = ""
        initialize_warp(self.device)

        self.arm_controlled_joints = [
            'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'
        ]
        self.hand_controlled_joints = [
            'index_joint_0', 'index_joint_1', 'index_joint_2', 'index_joint_3',
             'middle_joint_0', 'middle_joint_1', 'middle_joint_2', 'middle_joint_3',
             'ring_joint_0', 'ring_joint_1', 'ring_joint_2', 'ring_joint_3',
             'thumb_joint_0', 'thumb_joint_1', 'thumb_joint_2', 'thumb_joint_3'
         ]

        # Timestep for publishing fabric commands out
        self.rate_hz = 60.
        self.publish_dt = 1./self.rate_hz # sec
        # Timestep for integrating the fabric state
        self.fabric_dt = 1./self.rate_hz # sec
        
        # Declare position command signals for the arm PD controller and boolean for gripper
        self._kuka_joint_position_command_lock = Lock()
        self._allegro_joint_position_command_lock = Lock()
        self._kuka_joint_position_command = None
        self._kuka_joint_velocity_command = None
        self._allegro_joint_position_command = None
        self._allegro_joint_velocity_command = None

        # Declare measured feedback signals of robot
        self._kuka_joint_position_lock = Lock()
        self._allegro_joint_position_lock = Lock()
        self._kuka_joint_position = None
        self._allegro_joint_position = None

        # Declaration of commands going into fabric
        self._hand_target_lock = Lock()
        self._palm_target_lock = Lock()
        self.palm_target = None
        self.hand_target = None
        
        # Robot feedback health monitoring
        self.robot_synced = False
        self.kuka_feedback_time = time.time()
        self.allegro_feedback_time = time.time()
        self.robot_feedback_time_elapsed = 0. # sec
        self.heartbeat_time_threshold = .1 # sec

        # Set up pub/sub for kuka
        self._kuka_pub = self.create_publisher(JointState, "/kuka/joint_commands", 1)
        self._kuka_timer = self.create_timer(self.publish_dt, self._kuka_pub_callback)
        self._kuka_sub = self.create_subscription(
            JointState(),
            '/kuka/joint_states',
            self._kuka_sub_callback,
            1)

        # Set up pub/sub for allegro
        self._allegro_pub = self.create_publisher(JointState, "/allegro/joint_commands", 1)
        self._allegro_timer = self.create_timer(self.publish_dt, self._allegro_pub_callback)
        self._allegro_sub = self.create_subscription(
            JointState(),
            '/allegro/joint_states',
            self._allegro_sub_callback,
            1)

        # Set up sub for receiving commands for the fabric
        # Subscriber for getting pose commands
        self._kuka_allegro_pose_command_sub = self.create_subscription(
            JointState(),
            '/kuka_allegro_fabric/pose_commands',
            self._kuka_allegro_fabric_pose_command_sub_callback,
            1)

        # Subscriber for getting PCA commands
        self._kuka_allegro_pca_command_sub = self.create_subscription(
            JointState(),
            '/kuka_allegro_fabric/pca_commands',
            self._kuka_allegro_fabric_pca_command_sub_callback,
            1)

        # Set up publisher for broadcasting fabric state as feedback
        self._kuka_allegro_fabric_states_lock = Lock()
        self.kuka_allegro_fabric_states_msg = JointState()
        self.kuka_allegro_fabric_states_msg.name =\
            ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
             'index_joint_0', 'index_joint_1', 'index_joint_2', 'index_joint_3',
             'middle_joint_0', 'middle_joint_1', 'middle_joint_2', 'middle_joint_3',
             'ring_joint_0', 'ring_joint_1', 'ring_joint_2', 'ring_joint_3',
             'thumb_joint_0', 'thumb_joint_1', 'thumb_joint_2', 'thumb_joint_3']
        self._kuka_allegro_fabric_pub =\
            self.create_publisher(JointState, "/kuka_allegro_fabric/joint_states", 1)
        self._kuka_allegro_fabric_timer =\
            self.create_timer(self.publish_dt, self._kuka_allegro_fabric_pub_callback)

#    def _set_target_pose(self, pose):
#        msg = Pose()
#        msg.position.x = pose.p[0]
#        msg.position.y = pose.p[1]
#        msg.position.z = pose.p[2]
#        msg.orientation.w = pose.q[0]
#        msg.orientation.x = pose.q[1]
#        msg.orientation.y = pose.q[2]
#        msg.orientation.z = pose.q[3]
#        self._nex10_fabric_pose_command_sub_callback(msg)
#
#    def _set_cspace_position_target(self, posture_config):
#        # Get the posture config data into the following member. (Assigns line by line to bridge
#        # from numpy cpu to pytorch gpu. TODO: is there a better way to do this?)
#        with self.cspace_position_target_lock:
#            if self.cspace_position_target is not None:
#                for i in range(len(posture_config)):
#                    self.cspace_position_target[0,i] = posture_config[i]

    def _kuka_pub_callback(self):
        """
        Publishes latest kuka joint position command onto ROS 2 topic that
        goes to PD controller.
        """
        with self._kuka_joint_position_command_lock:
            if self._kuka_joint_position_command is not None and \
               self._kuka_joint_velocity_command is not None:
                msg = JointState()
                msg.name = self.arm_controlled_joints
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.position = self._kuka_joint_position_command
                msg.velocity = [0.] * 7 #self._kuka_joint_velocity_command
                #msg.velocity = self._kuka_joint_velocity_command
                msg.effort = []
                self._kuka_pub.publish(msg)
    
    def _kuka_sub_callback(self, msg):
        """
        Acquires the feedback time, sets the measured joint position for the
        kuka, and also sets the command for the kuka to this measured position
        if a command does not yet exist.
        ------------------------------------------
        :param msg: ROS 2 JointState message type
        """
        with self._kuka_joint_position_lock:
            self.kuka_feedback_time = time.time()
            self._kuka_joint_position = msg.position
        with self._kuka_joint_position_command_lock:
            if self._kuka_joint_position_command is None or self._kuka_joint_velocity_command is None:
                self._kuka_joint_position_command = msg.position
                self._kuka_joint_velocity_command = len(msg.velocity) * [0.]

    def _allegro_pub_callback(self):
        """
        Publishes latest allegro joint position command onto ROS 2 topic that
        goes to PD controller.
        """
        with self._allegro_joint_position_command_lock:
            if self._allegro_joint_position_command is not None and \
               self._allegro_joint_velocity_command is not None:
                msg = JointState()
                msg.name = self.hand_controlled_joints
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.position = self._allegro_joint_position_command
                msg.velocity = self._allegro_joint_velocity_command
                msg.effort = []
                self._allegro_pub.publish(msg)

    def _allegro_sub_callback(self, msg):
        """
        Acquires the feedback time, sets the measured joint position for the
        allegro, and also sets the command for the allegro to this measured position
        if a command does not yet exist.
        ------------------------------------------
        :param msg: ROS 2 JointState message type
        """
        with self._allegro_joint_position_lock:
            self.allegro_feedback_time = time.time()
            self._allegro_joint_position = msg.position
        with self._allegro_joint_position_command_lock:
            if self._allegro_joint_position_command is None or self._allegro_joint_velocity_command is None:
                self._allegro_joint_position_command = msg.position
                self._allegro_joint_velocity_command = len(msg.velocity) * [0.]
            
    def _kuka_allegro_fabric_pose_command_sub_callback(self, msg):
        """
        Sets the palm pose target coming in from the ROS topic.
        ------------------------------------------
        :param msg: ROS 2 JointState message type
        """
        with self._palm_target_lock:
            self.palm_target.copy_(torch.tensor([list(msg.position)], device=self.device))

    def _kuka_allegro_fabric_pca_command_sub_callback(self, msg):
        """
        Sets the PCA position target coming in from the ROS topic.
        ------------------------------------------
        :param msg: ROS 2 JointState message type
        """
        with self._hand_target_lock:
            self.hand_target.copy_(torch.tensor([list(msg.position)], device=self.device))
            
    def _kuka_allegro_fabric_pub_callback(self):
        """
        Writes out the full fabric state on the ROS topic.
        """
        with self._kuka_allegro_fabric_states_lock:
            if len(self.kuka_allegro_fabric_states_msg.position) > 0 and\
               len(self.kuka_allegro_fabric_states_msg.velocity) > 0:
                self.kuka_allegro_fabric_states_msg.header.stamp = self.get_clock().now().to_msg()
                self._kuka_allegro_fabric_pub.publish(self.kuka_allegro_fabric_states_msg)
    
    def robot_feedback_heartbeat(self):
        """
        Calculates the heartbeat to the real robot such that processes can be
        stopped if heartbeat not detected.
        """
        # Check to see if joint measurements are fresh.
        kuka_feedback_time_elapsed = None
        allegro_feedback_time_elapsed = None

        with self._kuka_joint_position_lock:
            kuka_feedback_time_elapsed = time.time() - self.kuka_feedback_time
        with self._allegro_joint_position_lock:
            allegro_feedback_time_elapsed = time.time() - self.allegro_feedback_time
        # Find the largest elapsed feedback time between the the arm and hand and
        # set that to the robot's feedback time
        if kuka_feedback_time_elapsed > allegro_feedback_time_elapsed:
            self.robot_feedback_time_elapsed = kuka_feedback_time_elapsed
        else:
            self.robot_feedback_time_elapsed = allegro_feedback_time_elapsed

    def set_joint_commands(self, q, qd, qdd):
        """
        Writes the fabric state as commands destined for PD control and
        also as feedback.
        ------------------------------------------
        :param q: bxn Numpy array of fabric position. b=batch_size, n=number of joints
        :param qd: bxn Numpy array of fabric velocity. b=batch_size, n=number of joints
        :param qdd: bxn Numpy array of fabric acceleration. b=batch_size, n=number of joints
        """
        # Set the fabric position as the command position for the robot
        with self._kuka_joint_position_command_lock:
            self._kuka_joint_position_command = list(q[0, :7])
            self._kuka_joint_velocity_command = list(qd[0, :7])
        with self._allegro_joint_position_command_lock:
            self._allegro_joint_position_command = list(q[0, 7:])
            self._allegro_joint_velocity_command = list(qd[0, 7:])

        # Pack the full fabric state into a ROS 2 JointState message as feedback
        with self._kuka_allegro_fabric_states_lock:
            self.kuka_allegro_fabric_states_msg.position = list(q[0, :])
            self.kuka_allegro_fabric_states_msg.velocity = list(qd[0, :])
            self.kuka_allegro_fabric_states_msg.effort = list(qdd[0, :])
            
    def run(self):
        # Initialize fabric-----------------------
        # Declare batch size and number of joints
        batch_size = 1
        num_dof = 23

        # Provide initial commands for the fabric
        # Palm pose target
        self.palm_target =\
            torch.tensor([[-0.6868,  0.0320,  0.685, -2.3873, -0.0824,  3.1301]], device=self.device)

        # Hand PCA target
        self.hand_target =\
            torch.tensor([[1.5, 1.5, 0., 0.5, -0.25]], device=self.device)

        # This creates a world model that book keeps all the meshes
        # in the world, their pose, name, etc.
        print('Importing world')
        world_filename = 'kuka_allegro_boxes'
        max_objects_per_env = 20
        world_model = WorldMeshesModel(batch_size=batch_size,
                                       max_objects_per_env=max_objects_per_env,
                                       device=self.device,
                                       world_filename=world_filename)

        # This reports back handles to the meshes which is consumed
        # by the fabric for collision avoidance
        object_ids, object_indicator = world_model.get_object_ids()

        # Create Kuka-Allegro pose-pca fabric
        kuka_allegro_fabric = KukaAllegroPoseFabric(
            batch_size, device=self.device, timestep=self.fabric_dt
        )

        # Create integrator for the fabric
        kuka_allegro_integrator = DisplacementIntegrator(kuka_allegro_fabric)
        
        # Allocate for fabric state
        q = torch.zeros(batch_size, num_dof, device=self.device)
        qd = torch.zeros(batch_size, num_dof, device=self.device)
        qdd = torch.zeros(batch_size, num_dof, device=self.device)

        # Create CUDA graph
        g = None
        q_new = None
        qd_new = None
        qdd_new = None
        with self._hand_target_lock:
            with self._palm_target_lock:
                # NOTE: elements of inputs must be in the same order as expected in the
                # set_features function of the fabric
                inputs = [self.hand_target, self.palm_target, "euler_zyx",
                          q.detach(), qd.detach(), object_ids, object_indicator]
                g, q_new, qd_new, qdd_new = capture_fabric(
                    kuka_allegro_fabric, q, qd, qdd, self.fabric_dt,
                    kuka_allegro_integrator, inputs, self.device)
                print('Captured fabric.')
        
        # Sleep a little to ensure feedback subs have received feedback
        time.sleep(self.heartbeat_time_threshold + 0.2)

        # Query feedback heartbeat
        self.robot_feedback_heartbeat()
        print(self.robot_feedback_time_elapsed)

        # Indicate if heartbeat time out.
        if self.robot_feedback_time_elapsed > self.heartbeat_time_threshold:
            print('Heartbeat timed out')

        # Main loop. While feedback from real robot meets heartbeat interval and ROS 2
        # has not been signaled for shutdown, keep cycling the fabric and publishing
        # its state as commands to the PD controller and as feedback
        control_iter = 0
        print_iter = 60
        loop_time_filtered = 0.
        while self.robot_feedback_time_elapsed < self.heartbeat_time_threshold and rclpy.ok():
            # First send robot joint commands to where it is now for about 1 second to get
            # communications flowing.
            if not self.robot_synced:
                # Copy over the kuka joint positions
                q[0, :7].copy_(torch.tensor(self._kuka_joint_position, device=self.device))
                # Copy over the allegro joint_positions
                q[0, 7:].copy_(torch.tensor(self._allegro_joint_position, device=self.device))

                # Set joint commands, which will be published over ROS
                self.set_joint_commands(q.detach().cpu().numpy().astype('float'), 
                                        qd.detach().cpu().numpy().astype('float'),
                                        qdd.detach().cpu().numpy().astype('float'))

                print('sending at curr')
                time.sleep(1.)

                self.robot_synced = True
            
            # Set start time
            start = time.time()

            # Set features/actions
            with self._hand_target_lock:
                with self._palm_target_lock:
                    # Integrate fabric forward
                    # NOTE: iters_per_cycle greater than 1 steps faster through
                    # the fabric integral curve
                    for i in range(self.iters_per_cycle):
                        # Replay the fabric graph
                        g.replay()
                        # Update states with graph output
                        q.copy_(q_new)
                        qd.copy_(qd_new)
                        qdd.copy_(qdd_new)

            # Set joint commands, which will be published over ROS
            self.set_joint_commands(q.detach().cpu().numpy().astype('float'), 
                                    qd.detach().cpu().numpy().astype('float'),
                                    qdd.detach().cpu().numpy().astype('float'))

            # Keep 30 Hz tick rate
            while (time.time() - start) < self.publish_dt:
                time.sleep(.00001)

            # Query feedback heartbeat
            self.robot_feedback_heartbeat()

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
            
if __name__ == '__main__':
    # Parse the fabrics speed mode
    parser = argparse.ArgumentParser()
    parser.add_argument('speed_mode', type=str,
                        help='Set the fabrics motion speed: normal, fast')
    args = None
    # Check to see the speed_mode argument was actually passed
    try:
        args = parser.parse_args()
    except:
        print('--------------------------')
        print('Ensure you set a speed mode')
        print('Please set it to "normal", "fast"')
        sys.exit()

    # Check for validity of set mode
    speed_mode = args.speed_mode
    speed_modes = ["normal", "fast"]
    if speed_mode not in speed_modes:
       print('Invalid speed mode. Please set it to "slow", "normal", "fast", or "superfast"')
       sys.exit()

    print("Starting Kuka Allegro fabric node")
    rclpy.init()

    # Create the fabric
    kuka_allegro_fabric_node = KukaAllegroFabricNode(speed_mode=speed_mode)

    # Spawn separate thread that spools the fabric
    spin_thread = Thread(target=rclpy.spin, args=(kuka_allegro_fabric_node,), daemon=True)
    spin_thread.start()

    time.sleep(1.)

    kuka_allegro_fabric_node.run()

    kuka_allegro_fabric_node.destroy_node()
    rclpy.shutdown()

    print('Fabric closed.')

