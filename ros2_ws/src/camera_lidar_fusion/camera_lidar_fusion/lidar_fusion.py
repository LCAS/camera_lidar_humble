import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, TimeSynchronizer

from cv_bridge import CvBridge
import numpy as np
from sensor_msgs_py import point_cloud2 as pc2

from scripts.fusser import Fusser


class SensorFusionNode(Node):

    def __init__(self):

        super().__init__('lidar_fusion')

        self.declare_parameter('image_subscriber', '/camera/image_raw')
        self.declare_parameter('calib_subscriber', '/camera/calibration')
        self.declare_parameter('lidar_subscriber', '/lidar/points')
        self.declare_parameter('image_publisher', '/lidar_fusion/result')
        self.declare_parameter('YOLO_model', 'yolov5su.pt')
        self.declare_parameter('technique', 'average')
        self.declare_parameter('reduction_factor', 0.9)
        self.declare_parameter('yolo_classes', [0, 2])

        image_sub = self.get_parameter('image_subscriber').value
        calib_sub = self.get_parameter('calib_subscriber').value
        lidar_sub = self.get_parameter('lidar_subscriber').value
        image_pub = self.get_parameter('image_publisher').value
        self.model = self.get_parameter('YOLO_model').value
        self.technique = self.get_parameter('technique').value
        self.RF = self.get_parameter('reduction_factor').value
        self.classes = self.get_parameter('yolo_classes').value

        self.create_subscription(String, calib_sub, self._calib_callback, 1)

        ts = TimeSynchronizer(
                [
                    Subscriber(self, Image, image_sub),
                    Subscriber(self, PointCloud2, lidar_sub)
                    ],
                10)

        ts.registerCallback(self._main_pipeline)

        self.publisher = self.create_publisher(Image, image_pub, 1)

        self.bridge = CvBridge()
        self.fusser = None

        self.get_logger().info("Nodo lidar_fusion iniciado y escuchando...")

    def _calib_callback(self, msg):

        if not self.fusser:

            for line in str(msg).split(r'\n'):

                if line.startswith('std_msgs'):
                    values = line.split("'")[1].split(':')[1].split()
                    P = np.array([float(val) for val in values])

                elif line.startswith('R_rect'):
                    values = [val for val in line.split()]
                    R0 = np.array([float(val) for val in values[1:]])

                elif line.startswith('Tr_velo_cam'):
                    values = [val for val in line.split()]
                    V2C = np.array([float(val) for val in values[1:]])


            P, R0, V2C = P.reshape(3, 4), R0.reshape(3, 3), V2C.reshape(3, 4)

            self.fusser = Fusser(
                P, R0, V2C, 
                self.model,
                self.technique,
                self.RF,
                self.classes)

    def _main_pipeline(self, image_msg, lidar_msg):

        if self.fusser:

            img = self.bridge.imgmsg_to_cv2(
                image_msg,
                desired_encoding='bgr8')

            points = pc2.read_points_numpy(
                lidar_msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True)

            final_img = self.fusser.pipeline(img.copy(), points)

            final_img_msg = self.bridge.cv2_to_imgmsg(
                final_img,
                encoding='bgr8')

            self.publisher.publish(final_img_msg)


def main(args=None):

    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
