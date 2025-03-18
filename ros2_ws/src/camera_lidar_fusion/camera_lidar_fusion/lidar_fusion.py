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

        self.declare_parameter('image_subscriber', '/default/sub')
        self.declare_parameter('calib_subscriber', '/default/sub')
        self.declare_parameter('lidar_subscriber', '/default/sub')

        image_sub = self.get_parameter('image_subscriber').value
        calib_sub = self.get_parameter('calib_subscriber').value
        lidar_sub = self.get_parameter('lidar_subscriber').value

        self.create_subscription(
                String,
                calib_sub,
                self.calibration_callback,
                10)

        ts = TimeSynchronizer(
                [
                    Subscriber(self, Image, image_sub),
                    Subscriber(self, PointCloud2, lidar_sub)
                    ],
                10)

        ts.registerCallback(self.process_data)

        self.declare_parameter('image_publisher', '/default/pub')

        image_pub = self.get_parameter('image_publisher').value

        self.publisher = self.create_publisher(
                Image,
                image_pub,
                1
                )

        self.bridge = CvBridge()
        self.fusser = None

        self.get_logger().info("Nodo lidar_fusion iniciado y escuchando...")

    def _image_raw2numpy(self, image_msg):

        return self.bridge.imgmsg_to_cv2(
                image_msg,
                desired_encoding="bgr8"
                )

    def _pointcloud2numpy(self, points_msg):

        return pc2.read_points_numpy(
                points_msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True)

    def _unpack_calib(self, msg):

        if not msg:
            return

        lines = str(msg).split(r'\n')

        for line in lines:

            if line.startswith('std_msgs'):
                values = line.split("'")[1].split(':')[1].split()
                P = np.array([float(val) for val in values])

            elif line.startswith('R_rect'):
                values = [val for val in line.split()]
                R0 = np.array([float(val) for val in values[1:]])

            elif line.startswith('Tr_velo_cam'):
                values = [val for val in line.split()]
                V2C = np.array([float(val) for val in values[1:]])

        return P.reshape(3, 4), R0.reshape(3, 3), V2C.reshape(3, 4)

    def calibration_callback(self, msg):

        if not self.fusser:
            P, R0, V2C = self._unpack_calib(msg)
            self.fusser = Fusser(P, R0, V2C)

    def process_data(self, image_msg, lidar_msg):

        if self.fusser:

            img = self._image_raw2numpy(image_msg)
            points = self._pointcloud2numpy(lidar_msg)

            final_result = self.fusser.pipeline(img.copy(), points)

            result_img_msg = self.bridge.cv2_to_imgmsg(
                    final_result,
                    encoding='bgr8'
                    )

            self.publisher.publish(result_img_msg)


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
