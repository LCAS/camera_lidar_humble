from setuptools import find_packages, setup

package_name = 'camera_lidar_fusion'
setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(
        include=[
            'camera_lidar_fusion',
            'camera_lidar_fusion.*',
            'scripts'],
        exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'numpy',
        'opencv-python'
        'matplotlib',
        'ultralytics',
        'sensor_msgs_py',
        ],
    extras_require={
        'test': ['pytest'],
    },
    zip_safe=True,
    maintainer='user',
    maintainer_email='ernestoroque777@gmail.com',
    description='Adaptation of the https://github.com/Vishalkagade/Camera-Lidar-Sensor-Fusion repository to a ROS2 Humble package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'lidar_fusion = camera_lidar_fusion.lidar_fusion:main',
        ],
    },
)
