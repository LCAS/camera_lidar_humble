import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, TimeSynchronizer

from cv_bridge import CvBridge
import numpy as np
from sensor_msgs_py import point_cloud2 as pc2
import csv

from scripts.fusser import Fusser
from my_msgs.msg import Float32MultiArrayStamped as F32M


class SensorFusionNode(Node):

    def __init__(self):

        super().__init__('lidar_fusion')

        self.declare_parameter('image_subscriber', '/camera/image_raw')
        self.declare_parameter('calib_subscriber', '/camera/calibration')
        self.declare_parameter('lidar_subscriber', '/lidar/points')
        self.declare_parameter('image_publisher', '/lidar_fusion/result')
        self.declare_parameter('image_fov_publisher', '/lidar_fusion/fov')
        self.declare_parameter('YOLO_model', 'yolov5su.pt')
        self.declare_parameter('technique', 'average')
        self.declare_parameter('reduction_factor', 0.9)
        self.declare_parameter('yolo_classes', [0, 2])
        self.declare_parameter('yolo_threshold', 0.8)
        self.declare_parameter('ped_bboxes_subscribers', '/detection_2d/pedestrian')
        self.declare_parameter('car_bboxes_subscribers', '/detection_2d/car')

        image_sub = self.get_parameter('image_subscriber').value
        calib_sub = self.get_parameter('calib_subscriber').value
        lidar_sub = self.get_parameter('lidar_subscriber').value
        image_pub = self.get_parameter('image_publisher').value
        image_lidar_pub = self.get_parameter('image_fov_publisher').value
        self.model = self.get_parameter('YOLO_model').value
        self.technique = self.get_parameter('technique').value
        self.RF = self.get_parameter('reduction_factor').value
        self.classes = self.get_parameter('yolo_classes').value
        self.yolo_threshold = self.get_parameter('yolo_threshold').value
        ped_boxes_sub = self.get_parameter('ped_bboxes_subscribers').value
        car_boxes_sub = self.get_parameter('car_bboxes_subscribers').value

        self.create_subscription(String, calib_sub, self._calib_callback, 1)

        ts = TimeSynchronizer(
                [
                    Subscriber(self, Image, image_sub),
                    Subscriber(self, PointCloud2, lidar_sub),
                    Subscriber(self, F32M, ped_boxes_sub),
                    Subscriber(self, F32M, car_boxes_sub)
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
                self.classes,
                self.yolo_threshold)

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

    def _get_frame(self, ped_boxes_msg, car_boxes_msg):
            ped_bboxes = np.array(ped_boxes_msg.data, dtype=np.float32)
            car_bboxes = np.array(car_boxes_msg.data, dtype=np.float32)

            n_ped = len(ped_bboxes) // 6
            n_car = len(car_bboxes) // 6

            if n_ped or n_car > 0:

                bboxes = np.append(ped_bboxes, car_bboxes)
                return bboxes[0]

            else:
                return None


    def _main_pipeline(self, image_msg, lidar_msg, ped_boxes_msg, car_boxes_msg):

        if self.fusser:

            img = self._imgmsg2np(image_msg)

            points = self._lidarmsg2np(lidar_msg)
            
            frame = self._get_frame(ped_boxes_msg, car_boxes_msg)

            _, final_img = self.fusser.pipeline(
                    img.copy(),
                    points,
                    frame)

            final_img_msg = self._np2imgmsg(final_img)

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
