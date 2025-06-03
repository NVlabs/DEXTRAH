import numpy as np
from scipy.spatial.transform import Rotation as R

# ROS imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import time

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('camera_tf_publisher')

    broadcaster = TransformBroadcaster(node)

    # Load in camera transform
    #T_robot_cam = np.loadtxt('robot_cam_left_calibration.txt', delimiter=',', dtype=float)
#    T_robot_cam =\
#        np.array([[9.979802254757542679e-01, 5.805126464282436838e-02, -2.579767882449228097e-02, -6.452117743594977251e-01],
#                  [2.867907587635045233e-02, -4.936231931159993508e-02, 9.983691061120923971e-01, -7.328016905360382749e-01],
#                  [5.668315593050039097e-02, -9.970924792142141779e-01, -5.092747518000630136e-02, 4.559887081479024329e-01],
#                  [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

    T_robot_cam =\
        np.array([[7.416679444534866883e-02,-9.902696855667120213e-01,1.177507386359286923e-01,-7.236400044878017468e-01],
[-1.274026398887237732e-01,1.076995435286611930e-01,9.859864987275952508e-01,-6.886495877727516479e-01],
[-9.890742408692511090e-01,-8.812921292808308105e-02,-1.181752422362273985e-01,6.366771698474239516e-01],
[0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])

    R_robot_cam = R.from_matrix(T_robot_cam[:3, :3])
    robot_cam_quat_xyzw = R_robot_cam.as_quat()

    while rclpy.ok():
        # Create a transform message
        transform = TransformStamped()
        transform.header.stamp = node.get_clock().now().to_msg()
        transform.header.frame_id = 'robot_base'
        transform.child_frame_id = 'camera2_depth_optical_frame'
        transform.transform.translation.x = T_robot_cam[0, 3]
        transform.transform.translation.y = T_robot_cam[1, 3]
        transform.transform.translation.z = T_robot_cam[2, 3]
        transform.transform.rotation.x = robot_cam_quat_xyzw[0]
        transform.transform.rotation.y = robot_cam_quat_xyzw[1]
        transform.transform.rotation.z = robot_cam_quat_xyzw[2]
        transform.transform.rotation.w = robot_cam_quat_xyzw[3]

        # Publish the transform
        broadcaster.sendTransform(transform)

        time.sleep(1./60)  # Publish at 1 Hz

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

