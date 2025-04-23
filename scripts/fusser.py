import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


class Fusser(object):

    def __init__(
            self,
            P, R0, V2C,
            model,
            RF,
            classes,
            thresh,
            points,
            bboxes,
            write):

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
        ''' If set, this method draw the bboxes of the detections '''

        x1, y1, x2, y2 = map(int, bbox[:4].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    def draw_points(self, img, original_indices):
        ''' If set, this method draw the lidar points contained in the
        bounding boxes of the detections '''

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
        ''' If set, this method writes the class and distances of the
        detections above the object '''

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
        ''' Filters out outliers from a tensor of 3D points (n, 3) using the
        modified z-score method based on the Median Absolute Deviation (MAD). '''

        median = points.median(dim=0).values
        mad = (points - median).abs().median(dim=0).values

        mad_safe = torch.where(mad == 0, torch.ones_like(mad) * 1e-6, mad)
        modified_z_scores = 0.6745 * (points - median) / mad_safe
        mask = (modified_z_scores.abs() < 3.5).all(dim=1)

        return points[mask]

    def get_coordinates(self, points):
        '''This method receives a distances array and return one
        distance depending on the chosen method'''

        if points.numel() != 0:
            coordinates = points.mean(dim=0)
            return torch.tensor([
                    -coordinates[1],
                    coordinates[2],
                    coordinates[0]])

    def run_obstacle_detection(self, img):
        '''This method runs a yolo model to make 2d detections on
        img'''

        predictions = self.model(
                img,
                conf=self.thresh,
                classes=self.classes,
                verbose=False)

        pred_bboxes = (predictions[0].boxes.data).detach()

        return pred_bboxes

    def get_lidar_on_image_fov(self, pc_velo, img):
        '''
        This method process the PointCluod and returns the
        original 3D points in the image (pc_velo) and the proyected
        points in the image (points_2d)
        '''

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

        global_mask = torch.zeros(len(self.points_2d), dtype=torch.bool).cuda()

        zeros = torch.zeros_like(pred_bboxes[:, :3])
        pred_bboxes = torch.cat([pred_bboxes, zeros], dim=1)

        for box in pred_bboxes:

            available_points = ~global_mask
            reduced_bbox = self.reduce_bbox(box)
            available_points_2d = self.points_2d[available_points]

            x_check = torch.logical_and(
                available_points_2d[:, 0] >= reduced_bbox[0],
                available_points_2d[:, 0] <= reduced_bbox[2]
            )

            y_check = torch.logical_and(
                available_points_2d[:, 1] >= reduced_bbox[1],
                available_points_2d[:, 1] <= reduced_bbox[3]
            )
            box_mask = torch.logical_and(x_check, y_check)

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
        ''' Main method of the class. Return the detections in a np array
        of the form (nx9), where we get (x1, y1, x2, y2, class, confidence, x, y, z)
        for each object '''

        self.get_lidar_on_image_fov(point_cloud[:, :3], image)
        pred_bboxes = self.run_obstacle_detection(image)

        if pred_bboxes.any():
            pred_bboxes = self.lidar_camera_fusion(pred_bboxes, image)

        return pred_bboxes.cpu().numpy(), image
