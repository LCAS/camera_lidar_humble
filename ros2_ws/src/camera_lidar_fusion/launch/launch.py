import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory


def generate_launch_description():
    config_file = os.path.join(
            get_package_share_directory("camera_lidar_fusion"),
            'config',
            'config.yaml'
            )

    return LaunchDescription([
        Node(
            package="camera_lidar_fusion",
            executable='lidar_fusion',
            name='lidar_fusion',
            parameters=[config_file]
            )
        ])
