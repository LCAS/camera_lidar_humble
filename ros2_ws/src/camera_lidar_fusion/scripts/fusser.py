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
            technique,
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

        self.technique = technique
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

        x1, y1, x2, y2 = map(int, bbox[:4].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    def draw_points(self, img, masked_depths, original_indices):

        selected_points = self.points_2d[original_indices]
        color_indices = torch.minimum(
            torch.floor(510.0 / masked_depths).to(torch.int32),
            torch.tensor(255, dtype=torch.int32)
        )

        pts_2d = selected_points.cpu().numpy().astype(int)
        color_indices = color_indices.cpu().numpy()

        for (x, y), color_idx in zip(pts_2d, color_indices):
            cv2.circle(img, (x, y), 1, tuple(self.cmap[color_idx]), -1)

    def write_distance(self, img, distance, box):

        x1, y1 = int(box[0]), int(box[1])
        class_ = 'car' if int(box[5]) == 2 else 'person'
        cv2.putText(
            img,
            f"{class_} : {distance:.2f}m",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    def filter_outliers(self, distances):

        median = np.median(distances)
        mad = np.median(np.abs(distances - median))

        if mad == 0 or np.isclose(mad, 0):
            modified_z_scores = np.zeros_like(distances)
        else:
            modified_z_scores = 0.6745 * (distances - median) / mad

        return distances[np.abs(modified_z_scores) < 3.5]

    def get_best_distance(self, distances):

        technique_map = {
            "closest": np.min,
            "average": np.mean,
            "random": lambda x: x[np.random.randint(len(x))],
            "median": np.median
        }

        return technique_map.get(self.technique, np.mean)(distances)

    def run_obstacle_detection(self, img):

        predictions = self.model(
                img,
                conf=self.thresh,
                classes=self.classes,
                verbose=False)

        pred_bboxes = (predictions[0].boxes.data).detach()

        return pred_bboxes

    def get_lidar_on_image_fov(self, pc_velo, img):

        pc_velo_torch = torch.tensor(pc_velo, dtype=torch.float32).cuda()
        ones = torch.ones(len(pc_velo_torch), 1, dtype=torch.float32).cuda()
        pts_3d_homo = torch.cat([pc_velo_torch, ones], dim=1)

        pts_2d_homo = torch.matmul(self.trafo_matrix, pts_3d_homo.T)
        pts_2d = (pts_2d_homo[:2] / pts_2d_homo[2]).T

        img_shape = torch.tensor(img.shape, dtype=torch.float32).cuda()

        fov_mask = (
            (pts_2d[:, 0] < img_shape[1]) &
            (pts_2d[:, 0] >= 0) &
            (pts_2d[:, 1] < img_shape[0]) &
            (pts_2d[:, 1] >= 0) &
            (pc_velo_torch[:, 0] > 2.0)
        )

        self.points_2d = pts_2d[fov_mask]
        self.pc_velo = pc_velo_torch[fov_mask]

    def lidar_camera_fusion(self, pred_bboxes, img):

        global_mask = torch.zeros(len(self.points_2d), dtype=torch.bool).cuda()

        zeros = torch.zeros_like(pred_bboxes[:, :1])
        pred_bboxes = torch.cat([pred_bboxes, zeros], dim=1)

        for box in pred_bboxes:
            box_center_x = (box[0] + box[2]) / 2
            box_center_y = (box[1] + box[3]) / 2
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]

            reduced_width = box_width * self.RF
            reduced_height = box_height * self.RF

            reduced_box = torch.stack([
                box_center_x - reduced_width / 2,
                box_center_y - reduced_height / 2,
                box_center_x + reduced_width / 2,
                box_center_y + reduced_height / 2
            ])

            available_points = ~global_mask

            if not torch.any(available_points):
                break

            available_points_2d = self.points_2d[available_points]

            x_check = torch.logical_and(
                available_points_2d[:, 0] >= reduced_box[0],
                available_points_2d[:, 0] <= reduced_box[2]
            )

            y_check = torch.logical_and(
                available_points_2d[:, 1] >= reduced_box[1],
                available_points_2d[:, 1] <= reduced_box[3]
            )
            box_mask = torch.logical_and(x_check, y_check)

            if not torch.any(box_mask):
                continue

            original_indices = torch.nonzero(available_points).squeeze()[box_mask]

            global_mask[original_indices] = True

            masked_depths = self.pc_velo[original_indices, 0]

            if self.points:
                self.draw_points(img, masked_depths, original_indices)
            if self.bboxes:
                self.draw_bboxes(img, box)

            if len(masked_depths) > 2:

                filtered_distances = self.filter_outliers(masked_depths.cpu().numpy())
                best_distance = self.get_best_distance(filtered_distances)

                if self.write:
                    self.write_distance(img, best_distance, box)

                box[-1] = float(best_distance)

        return pred_bboxes

    def pipeline(self, image, point_cloud):

        self.get_lidar_on_image_fov(point_cloud[:, :3], image)
        pred_bboxes = self.run_obstacle_detection(image)

        if pred_bboxes.any():
            pred_bboxes = self.lidar_camera_fusion(pred_bboxes, image)

        return pred_bboxes.cpu().numpy(), image
