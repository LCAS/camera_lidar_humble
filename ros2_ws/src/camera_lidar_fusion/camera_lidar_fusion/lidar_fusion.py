from time import time

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

        # Subscribers
        self.declare_parameter('image_sub_topic', '/camera/image_raw')
        self.declare_parameter('calib_sub_topic', '/camera/calibration')
        self.declare_parameter('lidar_sub_topic', '/lidar/points')
        image_sub_topic = self.get_parameter('image_sub_topic').value
        calib_sub_topic = self.get_parameter('calib_sub_topic').value
        lidar_sub_topic = self.get_parameter('lidar_sub_topic').value

        self.create_subscription(String, calib_sub_topic, self._calib_callback, 1)

        ts = TimeSynchronizer(
                [
                    Subscriber(self, Image, image_sub_topic),
                    Subscriber(self, PointCloud2, lidar_sub_topic),
                    ],
                10)

        ts.registerCallback(self._main_pipeline)

        # Publishers
        self.declare_parameter('result_pub_topic', '/camera_lidar_fusion/result')
        self.declare_parameter('bboxes_pub_topic', '/camera_lidar_fusion/pred_bboxes')
        result_pub_topic = self.get_parameter('result_pub_topic').value
        bboxes_pub_topic = self.get_parameter('bboxes_pub_topic').value

        self.final_publisher = self.create_publisher(Image, result_pub_topic, 1)
        self.bboxes_publisher = self.create_publisher(String, bboxes_pub_topic, 1)

        # Early fussion parameters
        self.declare_parameter('YOLO_model', 'yolov5su.pt')
        self.declare_parameter('distance_technique', 'average')
        self.declare_parameter('reduction_factor', 0.9)
        self.declare_parameter('yolo_classes', [0, 2])
        self.declare_parameter('yolo_threshold', 0.8)
        self.declare_parameter('draw_points', True)
        self.declare_parameter('draw_bboxes', True)
        self.declare_parameter('write_distance', True)

        # Node tools
        self.bridge = CvBridge()
        self.fusser = None

        self.get_logger().info("lidar_fusion node listening...")

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
                self.get_parameter('YOLO_model').value,
                self.get_parameter('distance_technique').value,
                self.get_parameter('reduction_factor').value,
                self.get_parameter('yolo_classes').value,
                self.get_parameter('yolo_threshold').value,
                self.get_parameter('draw_points').value,
                self.get_parameter('draw_bboxes').value,
                self.get_parameter('write_distance').value,
                )

    def _imgmsg2np(self, img_msg):
        return self.bridge.imgmsg_to_cv2(
                img_msg,
                desired_encoding='bgr8'
                )

    def _np2imgmsg(self, arr):
        return self.bridge.cv2_to_imgmsg(
                arr,
                encoding='bgr8'
                )

    def _lidarmsg2np(self, lidar_msg):
        return pc2.read_points_numpy(
                lidar_msg,
                field_names=('x', 'y', 'z'),
                skip_nans=True)

    def _np2String(self, arr):
        arr_string = np.array2string(arr)
        msg = String()
        msg.data = arr_string
        return msg

    def _main_pipeline(self, image_msg, lidar_msg):

        if self.fusser:

            img = self._imgmsg2np(image_msg)
            points = self._lidarmsg2np(lidar_msg)

            start = time()
            predictions, final_img = self.fusser.pipeline(
                    img,
                    points)

            final_img_msg = self._np2imgmsg(final_img)
            predictions_msg = self._np2String(predictions)

            self.final_publisher.publish(final_img_msg)
            self.bboxes_publisher.publish(predictions_msg)

            self.get_logger().info(f'{predictions.shape[0]} objects detected in {time()- start: .4f} secs')


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
