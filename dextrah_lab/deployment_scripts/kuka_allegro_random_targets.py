import os

import rclpy
from rclpy.node import Node
import threading
from sensor_msgs.msg import JointState

import numpy as np
import time
import copy
from threading import Thread, Lock
import torch


class KukaAllegroRandomTargets(Node):
    def __init__(self):
        super().__init__("kuka_allegro_random_commander")
        # Create mutex lock
        self.mutex = Lock()

        # Init node
        command_rate = 60. #0.5
        self.rate = self.create_rate(frequency=command_rate, clock=self.get_clock())

        # Some settings for cspace targets
        HAND_PCA_MINS = [0.2475, -0.3286, -0.7238, -0.0192, -0.5532]
        HAND_PCA_MAXS = [3.8336, 3.0025, 0.8977, 1.0243, 0.0629]
        PALM_POSE_MINS = [-1, -0.75, 0, -np.pi, -np.pi / 2, -np.pi]
        PALM_POSE_MAXS = [0.25, 0.75, 1, np.pi, np.pi / 2, np.pi]
        padding = 0.075
        self.upper_pose_limits = np.array(PALM_POSE_MAXS) - padding
        self.lower_pose_limits = np.array(PALM_POSE_MINS) + padding
        self.upper_pca_limits = np.array(HAND_PCA_MAXS) - padding
        self.lower_pca_limits = np.array(HAND_PCA_MINS) + padding

        # Joint command publisher
        self.pose_command_msg = JointState()
        self.pca_command_msg = JointState()
        self.kuka_allegro_fabric_pose_commands_pub = self.create_publisher(
            topic="/kuka_allegro_fabric/pose_commands",
            msg_type=JointState,
            qos_profile=1,
        )
        self.kuka_allegro_fabric_pca_commands_pub = self.create_publisher(
            topic="/kuka_allegro_fabric/pca_commands",
            msg_type=JointState,
            qos_profile=1,
        )

        self.nominal_pose =\
            np.array([-0.6868,  0.0320,  0.685, -2.3873, -0.0824,  3.1301])
            
        self.nominal_pca = np.array([1.5, 1.5, 0., 0.5, -0.25])

    def run(self):
        # While ROS is fine, keep sending random cspace targets
        action_step = 0
        action_switched = 0
        pose_command = copy.deepcopy(self.nominal_pose)
        pca_command = copy.deepcopy(self.nominal_pca)

        while rclpy.ok():
            # Generate target and send
#            pose_command = (
#                self.upper_pose_limits - self.lower_pose_limits
#            ) * np.random.rand(self.upper_pose_limits.shape[0]) + self.lower_pose_limits

            if action_step % 30 == 0:
                pose_command = copy.deepcopy(self.nominal_pose)
                if action_switched == 0:
                    #pose_command[1] -= .1
                    pose_command[2] -= .15
                    action_switched = 1
                else:
                    #pose_command[1] += .1
                    pose_command[2] += .15
                    action_switched = 0

                # pose_command = np.array(
                #     [-0.6868, 0.0320, 0.685, -2.3873, -0.0824, 3.1301]
                # )
#                pca_command = (
#                    self.upper_pca_limits - self.lower_pca_limits
#                ) * np.random.rand(self.upper_pca_limits.shape[0]) + self.lower_pca_limits
                pca_command = self.nominal_pca + np.random.rand(5)
                print(pose_command, pca_command)

            timestamp = self.get_clock().now().to_msg()
            self.pose_command_msg.header.stamp = timestamp
            self.pca_command_msg.header.stamp = timestamp
            self.pose_command_msg.position = pose_command.tolist()
            self.pca_command_msg.position = pca_command.tolist()

            print("Sending random command.")
            self.kuka_allegro_fabric_pose_commands_pub.publish(self.pose_command_msg)
            self.kuka_allegro_fabric_pca_commands_pub.publish(self.pca_command_msg)

            # Keep the 2 hz tick rate
            self.rate.sleep()

            action_step += 1


if __name__ == "__main__":
    rclpy.init()
    print("Sending random cspace position targets...")
    random_commander = KukaAllegroRandomTargets()
    thread = threading.Thread(target=rclpy.spin, args=(random_commander,), daemon=True)
    thread.start()
    time.sleep(1.)
    random_commander.run()

    random_commander.destroy_node()
    rclpy.shutdown()

