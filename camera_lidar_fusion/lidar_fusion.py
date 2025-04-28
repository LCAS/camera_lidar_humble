# Copyright 2025 Ernesto Roque: LCAS GROUP, University of Lincoln
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sensor Fusion Node (Camera and LiDAR)

This ROS 2 node subscribes to synchronized camera images and LiDAR point clouds,
fuses them using a YOLO-based early fusion approach, and publishes the
detection results and annotated images.

Author: Ernesto Roque: LCAS GROUP, University of Lincoln
License: Apache License 2.0
"""

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
    """Sensor Fusion Node combining 2D and 3D sensor data."""

    def __init__(self):
        """
        Initialize the SensorFusionNode.

        Sets up subscribers, publishers, parameters, and internal tools.
        """
        super().__init__('lidar_fusion')

        # Declare and get parameters
        self.declare_parameter('image_sub_topic', '/camera/image_raw')
        self.declare_parameter('calib_sub_topic', '/camera/calibration')
        self.declare_parameter('lidar_sub_topic', '/lidar/points')

        image_sub_topic = self.get_parameter('image_sub_topic').value
        calib_sub_topic = self.get_parameter('calib_sub_topic').value
        lidar_sub_topic = self.get_parameter('lidar_sub_topic').value

        # Subscribers
        self.create_subscription(String, calib_sub_topic, self._calib_callback, 1)
        ts = TimeSynchronizer(
            [
                Subscriber(self, Image, image_sub_topic),
                Subscriber(self, PointCloud2, lidar_sub_topic),
            ],
            queue_size=10
        )
        ts.registerCallback(self._main_pipeline)

        # Publishers
        self.declare_parameter('result_pub_topic', '/camera_lidar_fusion/result')
        self.declare_parameter('bboxes_pub_topic', '/camera_lidar_fusion/pred_bboxes')

        result_pub_topic = self.get_parameter('result_pub_topic').value
        bboxes_pub_topic = self.get_parameter('bboxes_pub_topic').value

        self.final_publisher = self.create_publisher(Image, result_pub_topic, 1)
        self.bboxes_publisher = self.create_publisher(String, bboxes_pub_topic, 1)

        # Early fusion parameters
        self.declare_parameter('YOLO_model', 'yolov5su.pt')
        self.declare_parameter('reduction_factor', 0.9)
        self.declare_parameter('yolo_classes', [0, 2])
        self.declare_parameter('yolo_threshold', 0.8)
        self.declare_parameter('draw_points', True)
        self.declare_parameter('draw_bboxes', True)
        self.declare_parameter('write_distance', True)

        # Node tools
        self.bridge = CvBridge()
        self.fusser = None

        # LiDAR ring settings
        self.declare_parameter('total_rings', 64)
        self.declare_parameter('rings_to_use', 64)
        self.total_rings = self.get_parameter('total_rings').value
        self.rings_to_use = self.get_parameter('rings_to_use').value
        self.ring_step = self.total_rings // self.rings_to_use

        self.get_logger().info("SensorFusionNode initialized and listening...")

    def _calib_callback(self, msg: String) -> None:
        """
        Parse calibration data from a String message and initialize the Fusser object.

        Args:
            msg (String): The calibration information.
        """
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
                self.get_parameter('reduction_factor').value,
                self.get_parameter('yolo_classes').value,
                self.get_parameter('yolo_threshold').value,
                self.get_parameter('draw_points').value,
                self.get_parameter('draw_bboxes').value,
                self.get_parameter('write_distance').value,
            )

    def _imgmsg2np(self, img_msg: Image) -> np.ndarray:
        """
        Convert a sensor_msgs/Image to a numpy array.

        Args:
            img_msg (Image): Image message.

        Returns:
            np.ndarray: OpenCV image.
        """
        return self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    def _np2imgmsg(self, arr: np.ndarray) -> Image:
        """
        Convert a numpy array to a sensor_msgs/Image message.

        Args:
            arr (np.ndarray): OpenCV image.

        Returns:
            Image: Image message.
        """
        return self.bridge.cv2_to_imgmsg(arr, encoding='bgr8')

    def _lidarmsg2np(self, lidar_msg: PointCloud2) -> np.ndarray:
        """
        Convert a sensor_msgs/PointCloud2 to a filtered numpy array.

        Args:
            lidar_msg (PointCloud2): PointCloud2 message.

        Returns:
            np.ndarray: Filtered point cloud array.
        """
        points = pc2.read_points_numpy(lidar_msg, field_names='xyz', skip_nans=True)

        points_per_ring = points.shape[0] // self.total_rings
        points = points[:points_per_ring * self.total_rings]
        points = points.reshape(self.total_rings, points_per_ring, 3)
        points = points[::self.ring_step]

        return points.reshape(self.rings_to_use * points_per_ring, 3)

    def _np2String(self, arr: np.ndarray) -> String:
        """
        Convert a numpy array to a std_msgs/String message.

        Args:
            arr (np.ndarray): Array to convert.

        Returns:
            String: String message.
        """
        arr_string = np.array2string(arr)
        msg = String()
        msg.data = arr_string
        return msg

    def _main_pipeline(self, image_msg: Image, lidar_msg: PointCloud2) -> None:
        """
        Main processing pipeline for synchronized camera and LiDAR messages.

        Args:
            image_msg (Image): Incoming camera image.
            lidar_msg (PointCloud2): Incoming LiDAR point cloud.
        """
        if self.fusser:
            img = self._imgmsg2np(image_msg)
            points = self._lidarmsg2np(lidar_msg)

            start = time()
            predictions, final_img = self.fusser.pipeline(img, points)
            spend = time() - start

            final_img_msg = self._np2imgmsg(final_img)
            predictions_msg = self._np2String(predictions)

            self.final_publisher.publish(final_img_msg)
            self.bboxes_publisher.publish(predictions_msg)

            self.get_logger().info(f'{len(predictions)} objects detected in {spend:.4f} seconds')


def main(args=None) -> None:
    """
    ROS 2 main entrypoint.
    """
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
