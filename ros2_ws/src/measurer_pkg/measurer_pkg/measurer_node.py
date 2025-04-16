import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from time import time


class Measurer(Node):

    def __init__(self):
        super().__init__('measurer_node')

        self.create_subscription(Image, '/camera_lidar_fusion/result', self._pred_fps, 1)
        self.create_subscription(Image, '/camera/image_raw', self._ground_fps, 1)
        self.create_subscription(String, '/finish/signal', self._finish, 1)

        self.ground = 0
        self.pred = 0

        self.start = None

        self.pred_file = open('/home/user/camera_lidar_humble/results/pred_fps.txt', 'w')
        self.ground_file = open('/home/user/camera_lidar_humble/results/ground_fps.txt', 'w')
        self.first = True

        self.get_logger().info("Measurer node online")

    def _finish(self, msg):
        self.destroy_node()
        rclpy.shutdown()

    def _pred_fps(self, pred_img):
        if self.start is None:
            self.start = time()

        if self.first:
            self.first = False
            return

        self.pred += 1
        fps = self.pred / (time() - self.start)
        self.pred_file.write(f'{fps}\n')
        self.get_logger().info("pred written")

    def _ground_fps(self, ground_img):
        if self.start is None:
            return

        self.ground += 1
        fps = self.ground / (time() - self.start)
        self.ground_file.write(f'{fps}\n')
        self.get_logger().info("ground written")

    def destroy_node(self):
        self.get_logger().info('Saving files and destroying node')
        self.pred_file.close()
        self.ground_file.close()

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Measurer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
