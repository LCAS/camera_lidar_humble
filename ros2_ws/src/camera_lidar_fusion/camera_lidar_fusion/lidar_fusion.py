import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, TimeSynchronizer

import cv_bridge
import numpy as np
from sensor_msgs_py import point_cloud2 as pc2

from scripts.fusser import Fusser

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_fusion')

        self.fuss = None

        self.create_subscription(
                String,
                '/camera/calibration',
                self.calibration_callback,
                10)

        self.publisher = self.create_publisher(
                Image,
                '/lidar_fusion/lidar_fov',
                1
                )

        image_sub = Subscriber(self, Image, '/camera/image_raw')
        lidar_sub = Subscriber(self, PointCloud2, '/lidar/points')

        ts = TimeSynchronizer([image_sub, lidar_sub], 100)
        ts.registerCallback(self.process_data)

        self.bridge = cv_bridge.CvBridge()

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
        if not self.fuss:
            P, R0, V2C = self._unpack_calib(msg)
            self.fuss = Fusser(P, R0, V2C)
            self.get_logger().info( "Fusser class online")

    def process_data(self, image_msg, lidar_msg):
        if self.fuss:

            img = self._image_raw2numpy(image_msg)
            points = self._pointcloud2numpy(lidar_msg)

            final_result = self.fuss.pipeline(img.copy(), points)

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

