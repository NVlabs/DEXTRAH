# Copyright (c) 2024, Nvidia.  All rights reserved.

import os
from threading import Lock, Thread
import math
import time
import argparse
import copy

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from tf2_msgs.msg import TFMessage
import tf2_ros

# Numpy and torch imports
import numpy as np
import torch

# Fabrics imports
from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.utils.rotation_utils import euler_to_matrix, matrix_to_euler, quaternion_to_matrix

# This class receives AR tag poses and joint position data to perform robot-camera calibration.
# Optimizer is Gauss-Newton. Robot-camera matrix is saved to txt file.
class OptimizeCameraCalibration():
    def __init__(self, camera):
        self.device = 'cpu'
        self.batch_size = 1

        self.timestep = 1./60

        # Initialize the fabric - we will use this to query forward kinematics
        self.kuka_allegro_fabric = KukaAllegroPoseFabric(self.batch_size,
                                                         self.device,
                                                         self.timestep,
                                                         graph_capturable=True
                                                         )

        self.camera = camera

    # Calculates the transform from camera to robot and palm_link to AR tag 
    # via optimization and the collected data.
    def calibrate_camera(self, tag_poses, robot_joints):
        # tag_poses are a list of 4x4 transform matrices
        # robot_joints are a list of 1d numpy arrays of size 7

        # Num iterations
        max_iters = 100
        min_iters = 30
        # Simple initialization of parameters, i.e., 0s
        parameters = np.array([0.] * 12)
        cost = 1000.
        cost_threshold = 10e-6
        for i in range(max_iters):
            # Early out if cost is sufficiently low
            if cost < cost_threshold and i > min_iters:
                break;
            # Perturb parameters if we detect optimization is stuck
            if (i % 10 == 0) and cost >= cost_threshold:
                print('scattering parameters')
                parameters = parameters + .5 * (np.random.rand(12) - .5)
            # Eval current cost
            cost, cost_vector = self.calc_pose_loss(parameters, tag_poses, robot_joints)
            print('Current cost', cost)
            #print('Current params', parameters)

            # Build numerical Jacobian
            jacobian = self.calc_jacobian(parameters, cost, cost_vector, tag_poses, robot_joints)

            # Run Gauss-Newton step
            parameters = self.gauss_newton(parameters, jacobian, cost_vector)

        return parameters

    def save_calibration_matrices(self, parameters, joint_angles):
        # Save calibration matrix to file.
        robot_T_cam = np.linalg.inv(self.transform_from_coordinates(parameters[:3], parameters[3:6]))
        print('Saved transform from robot to cam.')
        np.savetxt('robot_cam_' + self.camera + '_calibration.txt', robot_T_cam, delimiter=',')

#        # Calculating three total points for 6d pose loss
#        cam_T_robot = self.transform_from_coordinates(parameters[:3], parameters[3:6])
#        joint_angles_with_allegro = np.array(list(joint_angles) + [0.] * 16)
#        robot_T_palm = self.kuka_kinematics.pose(joint_angles_with_allegro, "palm_link").matrix()
#        cam_T_palm = np.dot(cam_T_robot, robot_T_palm)
#        
#        print('Saved transform from cam to palm')
#        np.savetxt('cam_palm_calibration.txt', cam_T_palm, delimiter=',')

    # Create 4x4 homogeneous transform from translation components and Euler angles.
    def transform_from_coordinates(self, translation, euler_angles):
        euler_angles = torch.tensor(euler_angles).unsqueeze(0).float()
        rotation_matrix = euler_to_matrix(euler_angles)[0]

        # Create homogeneous transform
        transform = np.zeros((4,4))
        transform[3,3] =  1.

        # Insert translation 
        transform[:3, 3] = translation

        # Insert rotation
        transform[:3, :3] = rotation_matrix

        return transform

    # Calculates loss of AR tag pose as measured between 3 noncollinear points.
    # Sums across all pose errors.
    def calc_pose_loss(self, parameters, tag_poses, robot_joints):
        # Iterate through all measured data.
        cost_vector = []
        for (meas_cam_T_tag, joint_angles) in zip(tag_poses, robot_joints):
            # Calculating three total points for 6d pose loss
            # Robot expressed in camera coordinates
            cam_T_robot = self.transform_from_coordinates(parameters[:3], parameters[3:6])

            # Gripper expressed in robot coordinates
            cspace_position_arm = torch.tensor(np.array([joint_angles]), device=self.device).float()
            cspace_hand = torch.zeros(1, 16, device=self.device)
            cspace_position = torch.cat((cspace_position_arm, cspace_hand), dim=-1)
            gripper_pose = self.kuka_allegro_fabric.get_palm_pose(cspace_position, "euler_zyx")

            rot_matrix = euler_to_matrix(gripper_pose[:, 3:])
            #robot_T_palm = gripper_transform[0,:,:].detach().cpu().numpy().astype(np.float64)
            robot_T_palm = np.eye(4)
            robot_T_palm[:3, :3] = rot_matrix[0,:,:].detach().cpu().numpy().astype(np.float64)
            robot_T_palm[:3, 3] = gripper_pose[0, :3].detach().cpu().numpy().astype(np.float64)

            # Tag expressed in gripper coordinates
            palm_T_tag = self.transform_from_coordinates(parameters[6:9], parameters[9:12])

            # Tag expressed in camera coordinates
            cam_T_palm = np.dot(cam_T_robot, robot_T_palm)
            cam_T_tag = np.dot(cam_T_palm, palm_T_tag)

            # First 3D point will be origin of transform
            first_point = cam_T_tag[:3, 3].flatten()

            # Second 3D point will have an x-coordinate offset from the origin.
            second_point_offset = np.zeros((4, 1))
            second_point_offset[0, 0] = 0.1 # change in x coordinate
            second_point_offset[3, 0] = 1.0 
            second_point_transform = np.dot(cam_T_tag, second_point_offset)
            second_point = second_point_transform[:3, 0]

            # Third 3D point will have an y-coordinate offset from the origin.
            third_point_offset = np.zeros((4, 1))
            third_point_offset[1, 0] = 0.1 # change in y coordinate
            third_point_offset[3, 0] = 1.0 
            third_point_transform = np.dot(cam_T_tag, third_point_offset)
            third_point = third_point_transform[:3, 0]

            # Pull out 3 3D points as measured by AR detection.
            first_point_target = meas_cam_T_tag[:3, 3].flatten()

            # Second 3D point will have an x-coordinate offset from the measured origin.
            second_point_target_transform = np.dot(meas_cam_T_tag, second_point_offset)
            second_point_target = second_point_target_transform[:3, 0]

            # Third 3D point will have an y-coordinate offset from the measured origin.
            third_point_target_transform = np.dot(meas_cam_T_tag, third_point_offset)
            third_point_target = third_point_target_transform[:3, 0]

            # Append translational errors to cost vector
            cost_vector += list(first_point - first_point_target)
            cost_vector += list(second_point - second_point_target)
            cost_vector += list(third_point - third_point_target)

        # Convert cost vector to numpy array and calculate quadratic cost.
        cost_vector = np.array(cost_vector)

        # Scale cost by number of data points as well.
        cost = (0.5 / len(tag_poses)) * np.dot(cost_vector, cost_vector)

        return (cost, cost_vector)

    # Calculates the Jacobian of the cost vector.
    def calc_jacobian(self, parameters, cost, cost_vector, tag_poses, robot_joints):
        jacobian = np.zeros((len(cost_vector), len(parameters)))
        eps = 1e-6
        # Old inefficient way of doing it
#        for i in range(len(cost_vector)):
#            for j in range(len(parameters)):
#                # Perturb param
#                parameters[j] += eps
#                # Eval new cost
#                cost_new, cost_vector_new = self.calc_pose_loss(parameters, tag_poses, robot_joints)
#
#                # Calculate element of Jacobian
#                jacobian[i,j] = (cost_vector_new[i] - cost_vector[i]) / eps
#
#                # Unperturb
#                parameters[j] -= eps

        # New, more efficient way of doing it
        for j in range(len(parameters)):
            # Perturb param
            parameters[j] += eps

            # Eval new cost
            cost_new, cost_vector_new = self.calc_pose_loss(parameters, tag_poses, robot_joints)

            # Calculate column of Jacobian
            for i in range(len(cost_vector)):
                jacobian[i,j] = (cost_vector_new[i] - cost_vector[i]) / eps

            # Unperturb
            parameters[j] -= eps

        return jacobian

    # Make Gauss-Newton update to parameters.
    def gauss_newton(self, parameters, jacobian, cost_vector):
        inertia = 1e-6 * np.eye(len(jacobian[0] ), len(jacobian[0]))

        # Search direction
        p = np.dot(np.dot(np.linalg.inv((np.dot(jacobian.transpose(), jacobian) + inertia)), jacobian.transpose()), cost_vector)
        alpha = 1. # Unit step

        parameters -= alpha * p

        return parameters

# This class will move robot, log pose data for camera calibration, perform
# camera calibration, and save results to file.
class CameraCalibrationNode(Node):
    def __init__(self, camera, x, y, z):
        # Initialize ROS
        rclpy.init()
        super().__init__('camera_calibration')
        
        self.device = 'cuda'
        self.batch_size = 1
        self.timestep = 1./60
        
        # Set the warp cache directory based on device
        warp_cache_dir = ""
        initialize_warp(self.device)

        # Initialize the fabric - we will use this to query forward kinematics
        self.kuka_allegro_fabric = KukaAllegroPoseFabric(self.batch_size,
                                                         self.device,
                                                         self.timestep,
                                                         graph_capturable=True
                                                         )

        # Timestep for publishing pose commands out
        self.publish_dt = 1./30

        # Declare measured robot joint positions
        self._kuka_joint_position = None

        # Data arrays. These will be list of numpy arrays
        self.tag_transforms = []
        self.robot_joints = []

        # Robot feedback health monitoring
        self.robot_synced = False
        self.kuka_feedback_time = time.time()
        self.robot_feedback_time_elapsed = None
        self.heartbeat_time_threshold = self.publish_dt + .02
        
        # TODO: remove the below uncommented code when no longer relevant
#        nominal_config =\
#            torch.tensor([[-1.046692, -0.3936, 0.83101, 1.79891, -0.91554, 0.46152, -1.97573,
#                          0., 0., 0., 0.,
#                          0., 0., 0., 0.,
#                          0., 0., 0., 0.,
#                          0., 0., 0., 0.]],
#                         device=self.device)
#        nom_pose = self.kuka_allegro_fabric.get_palm_pose(nominal_config, "euler_zyx")
#        input(nom_pose)

        # Robot gripper command pose publisher
        self._kuka_palm_pose_command = None
        self._kuka_pub = self.create_publisher(JointState, "/kuka_allegro_fabric/pose_commands", 1)
        self._kuka_timer = self.create_timer(self.publish_dt, self._kuka_pub_callback)

        # Robot joint position feedback
        self._kuka_sub = self.create_subscription(
            JointState(),
            '/kuka/joint_states',
            self._kuka_sub_callback,
            1)

        # Read fabric state. We will use this to detect once arm motion is sufficiently
        # small to take readings for camera calibration
        self._kuka_fabric_joint_velocity = None
        self._kuka_allegro_fabric_sub = self.create_subscription(
            JointState(),
            '/kuka_allegro_fabric/joint_states',
            self._kuka_allegro_fabric_sub_callback,
            1)
        
        # Camera tag subsriber
        #self.tf_buffer = tf2_ros.Buffer()
        #self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tag_sub = self.create_subscription(
            TFMessage(),
            '/tf',  # Replace with your TF topic
            self._tf_callback,
            10
        )
        self.got_tag_info = False
        self.tag_transform = np.zeros((4,4))
        self.tag_transform[3,3] = 1.

        # Optimizer
        self.optimizer = OptimizeCameraCalibration(camera)

        # Centrally located home pose
        # TODO: need to change the angles here. they are from nex10
        self.home_pose =\
            torch.tensor([[-0.6868,  0.0320,  0.685, -2.3873, -0.0824,  3.1301]],
                         device=self.device).expand((self.batch_size, 6)).float()

        # TODO: need to change the angles here. They are from nex10
        #gripper_target  = np.array([x, y, z, 0., 1.25, 0.])
        gripper_target =\
            torch.tensor([[-0.6621,  0.1553,  0.4997, -2.9915,  0.0843,  1.4322]],
                         device=self.device).expand((self.batch_size, 6)).float()

        euler_angles = gripper_target[:, 3:]
        rot_matrix = euler_to_matrix(euler_angles)[0]
        gripper_target_ref = torch.eye(4,4, device=self.device)
        gripper_target_ref[:3, 3] = gripper_target[0, :3]
        gripper_target_ref[:3, :3] = rot_matrix
        
        self.gripper_target_ref = gripper_target_ref.unsqueeze(0)

        # Create collection of offset poses
        # First number: translation along hand x
        # Second number: translation along hand y
        # Third number: translation along hand z
        # Fourth number: rotation about the hand x
        # Fifth number: rotation about the hand y
        # Sixth number: rotation about the hand z
        normalized_offsets = torch.tensor(
            [[0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.7500, 0.2500, 0.2500, 0.2500, 0.7500, 0.7500, 0.2500],
             [0.2500, 0.7500, 0.7500, 0.7500, 0.2500, 0.2500, 0.7500],
             [0.3750, 0.3750, 0.6250, 0.8750, 0.3750, 0.1250, 0.3750],
             [0.8750, 0.8750, 0.1250, 0.3750, 0.8750, 0.6250, 0.8750],
             [0.6250, 0.1250, 0.8750, 0.6250, 0.6250, 0.8750, 0.1250],
             [0.1250, 0.6250, 0.3750, 0.1250, 0.1250, 0.3750, 0.6250],
             [0.1875, 0.3125, 0.9375, 0.4375, 0.5625, 0.3125, 0.4375],
             [0.6875, 0.8125, 0.4375, 0.9375, 0.0625, 0.8125, 0.9375],
             [0.9375, 0.0625, 0.6875, 0.1875, 0.3125, 0.5625, 0.1875],
             [0.4375, 0.5625, 0.1875, 0.6875, 0.8125, 0.0625, 0.6875],
             [0.3125, 0.1875, 0.3125, 0.5625, 0.9375, 0.4375, 0.0625],
             [0.8125, 0.6875, 0.8125, 0.0625, 0.4375, 0.9375, 0.5625],
             [0.5625, 0.4375, 0.0625, 0.8125, 0.1875, 0.6875, 0.3125],
             [0.0625, 0.9375, 0.5625, 0.3125, 0.6875, 0.1875, 0.8125],
             [0.0938, 0.4688, 0.4688, 0.6562, 0.2812, 0.9688, 0.5312],
             [0.5938, 0.9688, 0.9688, 0.1562, 0.7812, 0.4688, 0.0312],
             [0.8438, 0.2188, 0.2188, 0.9062, 0.5312, 0.2188, 0.7812],
             [0.3438, 0.7188, 0.7188, 0.4062, 0.0312, 0.7188, 0.2812],
             [0.4688, 0.0938, 0.8438, 0.2812, 0.1562, 0.8438, 0.9062],
             [0.9688, 0.5938, 0.3438, 0.7812, 0.6562, 0.3438, 0.4062],
             [0.7188, 0.3438, 0.5938, 0.0312, 0.9062, 0.0938, 0.6562],
             [0.2188, 0.8438, 0.0938, 0.5312, 0.4062, 0.5938, 0.1562],
             [0.1562, 0.1562, 0.5312, 0.8438, 0.8438, 0.6562, 0.9688],
             [0.6562, 0.6562, 0.0312, 0.3438, 0.3438, 0.1562, 0.4688],
             [0.9062, 0.4062, 0.7812, 0.5938, 0.0938, 0.4062, 0.7188],
             [0.4062, 0.9062, 0.2812, 0.0938, 0.5938, 0.9062, 0.2188],
             [0.2812, 0.2812, 0.1562, 0.2188, 0.7188, 0.5312, 0.5938],
             [0.7812, 0.7812, 0.6562, 0.7188, 0.2188, 0.0312, 0.0938]], device=self.device)

        # Create list of post targets based on offsets above and nominal gripper pose
        self.pose_targets = []
        for i in range(normalized_offsets.shape[0]):
            self.pose_targets.append(self.calc_pose_from_offset(normalized_offsets[i,:]))

    def calc_pose_from_offset(self, offset_params):
        # Some scaling parameters
        trans_scaling = .05
        rot_scaling = torch.pi / 8.

        # Calculating offset transform
        offset_transform = torch.eye(4,4, device=self.device)
        offset_transform[0, 3] = 2. * (offset_params[0] - 0.5) * trans_scaling # x
        offset_transform[1, 3] = 2. * (offset_params[1] - 0.5) * trans_scaling # y
        offset_transform[2, 3] = 2. * (offset_params[2] - 0.5) * trans_scaling # z

        x_angle = 2. * (offset_params[3] - 0.5) * rot_scaling
        y_angle = 2. * (offset_params[4] - 0.5) * rot_scaling
        z_angle = 2. * (offset_params[5] - 0.5) * rot_scaling
        euler_angles = torch.tensor([[z_angle, y_angle, x_angle]], device=self.device)
        rot_matrix = euler_to_matrix(euler_angles)

        offset_transform[:3, :3] = rot_matrix

        offset_transform = offset_transform.unsqueeze(0)

        gripper_target_transform = torch.bmm(self.gripper_target_ref, offset_transform)

        # Convert gripper transform into (x,y,z,euler_zyx)
        gripper_target = torch.zeros(1, 6, device=self.device)
        gripper_target[0, :3] = gripper_target_transform[0, :3, 3]
        euler_angles =\
            matrix_to_euler(gripper_target_transform[:, :3, :3])
        gripper_target[0, 3:] = euler_angles

        gripper_target = gripper_target.float()

        return gripper_target

    def _kuka_pub_callback(self):
        """
        Publishes gripper pose commands to the kuka fabric node to control the robot
        """
        if self._kuka_palm_pose_command is not None:
            msg = JointState()
            msg.position = self._kuka_palm_pose_command
            self._kuka_pub.publish(msg)

    def _kuka_sub_callback(self, msg):
        """
        Acquires the feedback time, sets the measured joint position for the
        kuka, and also sets the command for the kuka to this measured position
        if a command does not yet exist.
        ------------------------------------------
        :param msg: ROS 2 JointState message type
        """
        self.kuka_feedback_time = time.time()
        self._kuka_joint_position = msg.position

    def _kuka_allegro_fabric_sub_callback(self, msg):
        self._kuka_fabric_joint_velocity = np.array(msg.velocity)[:7] # slice out arm angles

    def _tf_callback(self, msg):
        # Get the transform
        # NOTE: this assumes that only the AR tag transform is being published
        if len(msg.transforms) > 0:
            for tag in msg.transforms:
                if tag.child_frame_id == 'tag36h11:0':
                    # Pull out translation
                    trans = np.array([tag.transform.translation.x,
                                      tag.transform.translation.y,
                                      tag.transform.translation.z])

                    # Pull out rotation, convert to rotation matrix and add translation
#                    rot = np.array([tag.transform.rotation.x,
#                                    tag.transform.rotation.y,
#                                    tag.transform.rotation.z,
#                                    tag.transform.rotation.w])

                    quat = torch.tensor([[tag.transform.rotation.w,
                                          tag.transform.rotation.x,
                                          tag.transform.rotation.y,
                                          tag.transform.rotation.z]], device='cpu')
                                          
                    rot_matrix = quaternion_to_matrix(quat)[0].cpu().numpy()
                    self.tag_transform[:3, 3] = trans
                    self.tag_transform[:3, :3] = rot_matrix
                    self.got_tag_info = True
        else:
            self.got_tag_info = False

    # Add the current AR tag and joint position measurements to the data list.
    def record_data(self):
        # If tag data was received, then log tag pose and robot joints pair
        if self.got_tag_info is True:
            self.tag_transforms.append(copy.copy(self.tag_transform))
            self.robot_joints.append(np.array(self._kuka_joint_position))
        else:
            print('Failed to get AR tag pose!!!')

        # Set tag info back to false
        self.got_tag_info = False

    def move_to_pose_target(self, pose_target, move_timeout, fabric_vel_threshold):
        start = time.time()

        # Command set which will auto issue over ROS 2 topic
        self._kuka_palm_pose_command =\
            list(pose_target[0,:].detach().cpu().numpy().astype('float'))

        # Monitor progress until sufficient accuracy achieved or timed out
        while (time.time() - start) < move_timeout and rclpy.ok():
            # Check heartbeat on robot feedback:
            if (time.time() - self.kuka_feedback_time) > .1:
                print("No feedback with robot. Killing process-----------------------------")
                rclpy.shutdown()

            # Early out if motion is sufficiently small
            if np.linalg.norm(self._kuka_fabric_joint_velocity) < fabric_vel_threshold and\
               (time.time() - start) > 1.:
                break

            time.sleep(.1)

    # Main method for moving the robot, logging data, and performing calibration.
    def run(self):
        # Move robot to home pose
        move_timeout = 10. # seconds
        fabric_vel_threshold = .05
        reach_tolerance = 0.01
        #input('hit ENTER when ready to move to home pose')
        print('Moving to home pose')
        self.move_to_pose_target(self.home_pose, move_timeout, fabric_vel_threshold)
        if not rclpy.ok():
            return

        # Cycle through the various pose targets, collecting data
        pose_index = 0
        for pose_target in self.pose_targets:
            # Move robot to next pose target
            # This blocks until timed out or pose is accurately
            # reached (whichever comes first)
            #print('pose target', pose_target)
            print('pose index', pose_index)
            if pose_index == 0:
                # Longer time for first move
                self.move_to_pose_target(pose_target, move_timeout, fabric_vel_threshold)
            else:
                # Faster time for subsequent moves
                self.move_to_pose_target(pose_target, move_timeout, fabric_vel_threshold)

            pose_index += 1

            if not rclpy.ok():
                return

            # Record robot joint and tag pose data.
            time.sleep(.5) # Small sleep to get data

            # Record camera pose and joint angles
            self.record_data()
        
        # Return robot to home pose.
        print('Moving to home pose')
        self.move_to_pose_target(self.home_pose, move_timeout, fabric_vel_threshold)
        if not rclpy.ok():
            return

        # Run optimizer on collected data.
        print('Running optimizer...')
        parameters = self.optimizer.calibrate_camera(self.tag_transforms, self.robot_joints)

        # Save calibration matrices given optimized parameters and last joint angle reading which
        # places the robot in a particular joint configuration for cube manipulation.
        print('Saving calibration matrices')
        self.optimizer.save_calibration_matrices(parameters, None)
        print('Done!')

if __name__ == '__main__':
    '''
    Run with:
    python camera_calibration.py --camera=left --x=0 --y=0 --z=0
    '''
    # Get which camera to target for calibration
    parser = argparse.ArgumentParser(description='Camera calibration script.')
    parser.add_argument('--camera', type=str, help='Specify target camera for calibration: "right", "left", or "center"')
    parser.add_argument('--x', type=float, help='Specify x param')
    parser.add_argument('--y', type=float, help='Specify y param')
    parser.add_argument('--z', type=float, help='Specify z param')
    args = parser.parse_args()
    camera = args.camera
    x = args.x
    y = args.y
    z = args.z
    
    if camera != "right" and camera != "left" and camera != "center":
        raise ValueError('Incorrect camera specification. Use either "right", "left", or "center"')

    # Instantiate camera calibration object.
    cam_calib_node = CameraCalibrationNode(camera=camera, x=x, y=y, z=z)

    # Spawn separate thread that handles ros spinning
    spin_thread = Thread(target=rclpy.spin, args=(cam_calib_node,), daemon=True)
    spin_thread.start()

    # Run the main control thread
    cam_calib_node.run()

    # Shutdown
    cam_calib_node.destroy_node()
    rclpy.shutdown()

    print('Camera calibration node closed.')

