# **ROS2 Humble Camera-LiDAR Fusion**  

This repository is an adaptation of [this repository](https://github.com/Vishalkagade/Camera-Lidar-Sensor-Fusion) by **Vishalkagade**, modified to work as a **ROS 2 Humble Hawksbill** package. The main changes involve acquiring data through **ROS 2 topics** instead of specific files, enabling **real-time processing**.  

Additionally, the computational approach has been optimized to achieve real-time execution. **PyTorch** is utilized to speed up point cloud processing from the LiDAR, and all required functions have been consolidated into a single script called `fusser.py`.  

---

## **Pipeline**  

Despite the modifications, this package retains most of the original code. For additional details, please refer to the [original repository](https://github.com/Vishalkagade/Camera-Lidar-Sensor-Fusion).  

The key contribution of this adaptation is the integration with **ROS 2**. The communication flow is structured as follows:  

![Pipeline of the node](readme_files/pipeline.png)

---
## **Package Overview**  

- **Package name:** `camera_lidar_fusion`  
- **Node name:** `lidar_fusion`  
- **Required inputs:**  
  - **Camera calibration parameters**  
  - **Image from the camera**  
  - **Point cloud from the LiDAR**
- **Outputs:**
  - **Image with the detections**
  - **String message with the detections**

## **Configuration**

This node has a launch file which loads the configuration params from a `config.yaml` file in the `config` directory inside the package. The following parameters can be configured:

- `image_sub_topic`: The topic where the raw images from the camera are published.
- `calib_sub_topic`: The topic where the calibration data are being published.
- `lidar_sub_topic`: The topic where the point cloud is being published.
- `result_pub_topic`: The topic where the output image will be published.
- `bboxes_pub_topic`: The topic where the detections will be published.
- `YOLO_model`: The YOLO model to be used. It's intended to be used with YOLOv5, but you can try with another one if needed.
- `reduction_factor`: The reduction factor of the segmented cloud in each bounding box, in order to reduce outliers.
- `yolo_classes`: The classes the model will be detecting according with COCO.
- `yolo_threshold`: The threshold that YOLO will use for classifying inferences.
- `draw_points`: If the output image should show the points of the lidar for each object.
- `draw_bboxes`: If the output image should show the bounding boxes of each object.
- `write_distance`: If the output image should show the distance to each object.
- `total_rings`: Many LiDARs don't publish the number of rings. Here you have to specify how many rings your LiDAR has.
- `rings_to_use`: Since you would like to reduce the computational work or try with different conditions, here you have to specify how many rings should the algorithm use.

## **Requisites**

- **UBUNTU 22.04**
- **CUDA capable GPU and CUDA drivers**
- **ROS2 Humble Hawksbill**

## **Installation and usage**

To set up and launch this package, follow these steps:  

1. **Create a new workspace and move in:**  
   ```bash
   mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
   ```

2. **Clone this repository:**  
   ```bash
   git clone git@github.com:LCAS/camera_lidar_humble.git
   ```

3. **Install the required libraries:**  
   ```bash
   rosdep init && rosdep update && rosdep install --from-paths .
   pip install -r requirements.txt
   ```

4. **Build the package and source the workspace:**  
   ```bash
   cd ~/ros2_ws && colcon build --packages-select camera_lidar_fusion
   source ~/ros2_ws/install/setup.bash
   ```

5. **Launch the node:**  
   ```bash
   ros2 launch camera_lidar_fusion launch.py
   ```

Ensure that the necessary camera and LiDAR topics are active before launching the node.  

---

## **Contributions & Issues**  

Feel free to contribute to this project or report any issues. Fork the repository and submit a **pull request** for improvements.  
