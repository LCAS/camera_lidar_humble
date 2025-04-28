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
Fusser: Camera-LiDAR Early Fusion Module

This module implements the early fusion of 2D object detections (YOLO)
with 3D LiDAR points, allowing the assignment of 3D coordinates to detected objects.

Author: Ernesto Roque: LCAS GROUP, University of Lincoln
License: Apache License 2.0
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


class Fusser:
    """Early fusion between camera detections and LiDAR point cloud data."""

    def __init__(self, P, R0, V2C, model, RF, classes, thresh, points, bboxes, write):
        """
        Initialize the Fusser.

        Args:
            P (np.ndarray): Camera projection matrix (3x4).
            R0 (np.ndarray): Rectification rotation matrix (3x3).
            V2C (np.ndarray): Velodyne-to-Camera transformation matrix (3x4).
            model (str): Path to YOLO model.
            RF (float): Reduction factor for bounding boxes.
            classes (list): Classes to detect.
            thresh (float): Confidence threshold.
            points (bool): Whether to draw LiDAR points.
            bboxes (bool): Whether to draw bounding boxes.
            write (bool): Whether to write coordinates on image.
        """
        R0_padded = np.column_stack([np.vstack([R0, [0, 0, 0]]), [0, 0, 0, 1]])
        V2C_padded = np.vstack((V2C, [0, 0, 0, 1]))
        trafo_matrix_np = np.dot(P, np.dot(R0_padded, V2C_padded))
        self.trafo_matrix = torch.tensor(trafo_matrix_np, dtype=torch.float32).cuda()

        self.RF = RF
        self.classes = classes
        self.thresh = thresh
        self.points = points
        self.bboxes = bboxes
        self.write = write

        self.model = YOLO(model).to(torch.device('cuda'))

        cmap = plt.cm.get_cmap("hsv", 256)
        self.cmap = cmap(np.arange(256))[:, :3] * 255

    def draw_bboxes(self, img, bbox):
        """Draws bounding boxes on the image."""
        x1, y1, x2, y2 = map(int, bbox[:4].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    def draw_points(self, img, original_indices):
        """Draws the LiDAR points associated with a detection on the image."""
        selected_points = self.points_2d[original_indices]
        color_indices = torch.minimum(
            torch.floor(510.0 / self.pc_velo[original_indices, 0]).to(torch.int32),
            torch.tensor(255, dtype=torch.int32)
        )

        pts_2d = selected_points.cpu().numpy().astype(int)
        color_indices = color_indices.cpu().numpy()

        for (x, y), color_idx in zip(pts_2d, color_indices):
            cv2.circle(img, (x, y), 1, tuple(self.cmap[color_idx]), -1)

    def write_coordinates(self, img, coordinates, box):
        """Writes class label and 3D coordinates above the detected object."""
        x1, y1 = int(box[0]), int(box[1])
        x, y, z = map(float, coordinates)

        class_ = 'car' if int(box[5]) == 2 else 'person'
        cv2.putText(
            img,
            f"{class_} : {x:.2f},{y:.2f},{z:.2f}m",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    def filter_outliers(self, points: torch.Tensor) -> torch.Tensor:
        """
        Removes outliers from a set of points using the Modified Z-Score method.

        Args:
            points (torch.Tensor): Input points (n, 3).

        Returns:
            torch.Tensor: Filtered points.
        """
        median = points.median(dim=0).values
        mad = (points - median).abs().median(dim=0).values

        mad_safe = torch.where(mad == 0, torch.ones_like(mad) * 1e-6, mad)
        modified_z_scores = 0.6745 * (points - median) / mad_safe
        mask = (modified_z_scores.abs() < 3.5).all(dim=1)

        return points[mask]

    def get_coordinates(self, points):
        """
        Calculates the mean 3D coordinate of a set of points.

        Args:
            points (torch.Tensor): Input points (n, 3).

        Returns:
            torch.Tensor: Mean coordinate (3D).
        """
        if points.numel() != 0:
            coordinates = points.mean(dim=0)
            return torch.tensor([-coordinates[1], coordinates[2], coordinates[0]])

    def run_obstacle_detection(self, img):
        """
        Runs YOLO model on the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            torch.Tensor: Bounding boxes (n, 6).
        """
        predictions = self.model(img, conf=self.thresh, classes=self.classes, verbose=False)
        pred_bboxes = (predictions[0].boxes.data).detach()
        return pred_bboxes

    def get_lidar_on_image_fov(self, pc_velo, img):
        """
        Projects LiDAR points onto the image plane and selects points within FOV.

        Args:
            pc_velo (np.ndarray): Point cloud (n, 3).
            img (np.ndarray): Input image.
        """
        pc_velo_torch = torch.tensor(pc_velo, dtype=torch.float32).cuda()
        ones = torch.ones(len(pc_velo_torch), 1, dtype=torch.float32).cuda()
        pts_3d_homo = torch.cat([pc_velo_torch, ones], dim=1)

        pts_2d_homo = torch.matmul(self.trafo_matrix, pts_3d_homo.T)
        pts_2d = (pts_2d_homo[:2] / pts_2d_homo[2]).T

        fov_mask = (
            (pts_2d[:, 0] < img.shape[1]) &
            (pts_2d[:, 0] >= 0) &
            (pts_2d[:, 1] < img.shape[0]) &
            (pts_2d[:, 1] >= 0) &
            (pc_velo_torch[:, 0] > 2.0)
        )

        self.points_2d = pts_2d[fov_mask]
        self.pc_velo = pc_velo_torch[fov_mask]

    def reduce_bbox(self, box):
        """
        Shrinks a bounding box based on the reduction factor (RF).

        Args:
            box (torch.Tensor): Bounding box (4D).

        Returns:
            torch.Tensor: Reduced bounding box.
        """
        box_center_x = (box[0] + box[2]) / 2
        box_center_y = (box[1] + box[3]) / 2
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]

        reduced_width = box_width * self.RF
        reduced_height = box_height * self.RF

        return torch.stack([
            box_center_x - reduced_width / 2,
            box_center_y - reduced_height / 2,
            box_center_x + reduced_width / 2,
            box_center_y + reduced_height / 2
        ])

    def lidar_camera_fusion(self, pred_bboxes, img):
        """
        Associates 2D detections with 3D LiDAR points.

        Args:
            pred_bboxes (torch.Tensor): 2D detections.
            img (np.ndarray): Input image.

        Returns:
            torch.Tensor: Updated detections with 3D coordinates.
        """
        global_mask = torch.zeros(len(self.points_2d), dtype=torch.bool).cuda()
        zeros = torch.zeros_like(pred_bboxes[:, :3])
        pred_bboxes = torch.cat([pred_bboxes, zeros], dim=1)

        for box in pred_bboxes:
            available_points = ~global_mask
            reduced_bbox = self.reduce_bbox(box)
            available_points_2d = self.points_2d[available_points]

            x_check = (available_points_2d[:, 0] >= reduced_bbox[0]) & (available_points_2d[:, 0] <= reduced_bbox[2])
            y_check = (available_points_2d[:, 1] >= reduced_bbox[1]) & (available_points_2d[:, 1] <= reduced_bbox[3])
            box_mask = x_check & y_check

            if not torch.any(box_mask):
                continue

            original_indices = torch.nonzero(available_points).squeeze()[box_mask]
            global_mask[original_indices] = True

            masked_points = self.pc_velo[original_indices, :]

            if self.points:
                self.draw_points(img, original_indices)
            if self.bboxes:
                self.draw_bboxes(img, box)

            if len(masked_points) > 2:
                filtered_points = self.filter_outliers(masked_points)
                coordinates = self.get_coordinates(filtered_points)
                if coordinates is None:
                    continue

                if self.write:
                    self.write_coordinates(img, coordinates, box)

                box[-3:] = coordinates

        return pred_bboxes

    def pipeline(self, image, point_cloud):
        """
        Full processing pipeline: detection, fusion, and output formatting.

        Args:
            image (np.ndarray): Input image.
            point_cloud (np.ndarray): Input point cloud.

        Returns:
            np.ndarray: Detections (n, 9) [x1, y1, x2, y2, class, confidence, x, y, z].
            np.ndarray: Annotated image.
        """
        self.get_lidar_on_image_fov(point_cloud[:, :3], image)
        pred_bboxes = self.run_obstacle_detection(image)

        if pred_bboxes.any():
            pred_bboxes = self.lidar_camera_fusion(pred_bboxes, image)

        return pred_bboxes.cpu().numpy(), image
