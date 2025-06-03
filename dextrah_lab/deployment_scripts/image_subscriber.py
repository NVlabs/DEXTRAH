import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

import matplotlib.pyplot as plt

WIDTH = 640
HEIGHT = 480

def object_image_to_tensor(msg):
    '''
    msg: integer values 0-255 (encoding: '16UC1' or 'mono16')
         expected resolution: 640x480
    The learned model expects real values between -0.5 and -1.3.
    '''
    img_np = np.frombuffer(msg.data, dtype=np.uint16).reshape(HEIGHT, WIDTH).astype(np.float32)
    img_np = cv2.resize(img_np, (WIDTH//4, HEIGHT//4), interpolation=cv2.INTER_LINEAR)
    #cv2.imwrite("output_image_unprocessed.png", img_np_to_save)
    #input("unprocessed image saved")
    img_np *= -1e-3
    img_np[img_np < -1.3] = 0
    img_np[img_np > -0.5] = 0
    return img_np

    #with open('img_np.npy', 'wb') as f:
    #    np.save(f, img_np)

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/robot2/camera2/aligned_depth_to_color/image_raw',  
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.fig = plt.figure()
        x = np.linspace(0,50.,num=WIDTH//4)
        y = np.linspace(0,50.,num=HEIGHT//4)
        X,Y = np.meshgrid(x,y)
        self.ax = self.fig.add_subplot(1,1,1)
        self.rendered_img = self.ax.imshow(X, vmin=-1.3,vmax=0, cmap='Greys')
        self.fig.canvas.draw()
        plt.title("Input")
        plt.show(block=False)
        print('here')

    def listener_callback(self, msg):
        self.get_logger().info('Received an image')
        image = object_image_to_tensor(msg)
        self.rendered_img.set_data(image)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
