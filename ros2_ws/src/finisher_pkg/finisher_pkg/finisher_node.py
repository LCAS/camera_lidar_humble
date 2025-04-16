import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class OneShotPublisher(Node):
    def __init__(self):
        super().__init__('finisher_node')
        self.publisher_ = self.create_publisher(String, '/finish/signal', 10)
        
        # Publicar una vez después de 0.5 segundos
        self.timer = self.create_timer(0.5, self.publish_and_shutdown)

    def publish_and_shutdown(self):
        msg = String()
        msg.data = 'Stop all'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Finish signal sent')
        
        # Cancelar el timer y cerrar el nodo
        self.timer.cancel()
        super().destroy_node()
        exit()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = OneShotPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
