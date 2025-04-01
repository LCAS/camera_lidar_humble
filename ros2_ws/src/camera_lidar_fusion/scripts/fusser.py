import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

import csv


class Fusser(object):

    def __init__(self, P, R0, V2C, model, technique, RF, classes, thresh):

        self.P = P
        self.V2C = V2C
        self.R0 = R0

        self.technique = technique
        self.RF = RF
        self.classes = classes
        self.thresh = thresh

        if torch.cuda.is_available():
            self.model = YOLO(model).to(torch.device('cuda'))
        else:
            self.model = YOLO(model)

    def filter_outliers(distances):
        median = np.median(distances)
        mad = np.median(np.abs(distances - median))
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

    def run_obstacle_detection(self, img, frame):


        predictions = self.model(
                img,
                conf=self.thresh,
                classes=self.classes,
                verbose=False)

        csv_file = '/home/user/pred_bboxes.csv'

        for r in predictions:
            pred_bboxes = (r.boxes.data).detach().cpu().numpy()

            if frame:

                with open(csv_file, mode='a',newline='') as file:
                    writer = csv.writer(file)
                    
                    for bbox in pred_bboxes:
                        x1, y1, x2, y2, conf, class_id = bbox
                        writer.writerow([frame, int(class_id), conf, x1, y1, x2, y2])

            result = r.plot()

        return result, pred_bboxes

    def get_lidar_on_image_fov(self, pc_velo, img):
        trafo_matrix = np.dot(self.P, np.dot(
           np.column_stack([np.vstack([self.R0, [0, 0, 0]]), [0, 0, 0, 1]]),
           np.vstack((self.V2C, [0, 0, 0, 1]))
        ))

        pts_3d_homo = np.column_stack([pc_velo, np.ones(len(pc_velo))])
        pts_2d_homo = np.dot(trafo_matrix, pts_3d_homo.T)
        pts_2d = (pts_2d_homo[:2] / pts_2d_homo[2]).T

        fov_mask = (
            (pts_2d[:, 0] < img.shape[1]) &
            (pts_2d[:, 0] >= 0) &
            (pts_2d[:, 1] < img.shape[0]) &
            (pts_2d[:, 1] >= 0) &
            (pc_velo[:, 0] > 2.0)
        )

        self.imgfov_pc_velo = pc_velo[fov_mask]
        self.imgfov_pts_2d = pts_2d[fov_mask, :]

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        img2 = img.copy()
        
        for i in range(self.imgfov_pts_2d.shape[0]):
            depth = self.imgfov_pc_velo[i,0]
            color = cmap[int(510.0 / depth), :]
            cv2.circle(
                img2,
                (int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))),
                2,
                color=tuple(color),
                thickness=-1,
            )

        return img2

    def lidar_camera_fusion(self, pred_bboxes, img):
        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = cmap(np.arange(256))[:, :3] * 255


        points_2d = torch.tensor(self.imgfov_pts_2d, dtype=torch.float32).cuda()
        pc_velo = torch.tensor(self.imgfov_pc_velo, dtype=torch.float32).cuda()

        global_mask = torch.zeros(len(points_2d), dtype=torch.bool).cuda()

        for box in pred_bboxes:
            box_tensor = torch.tensor(box, dtype=torch.float32).cuda()

            box_center_x = (box_tensor[0] + box_tensor[2]) / 2
            box_center_y = (box_tensor[1] + box_tensor[3]) / 2
            box_width = box_tensor[2] - box_tensor[0]
            box_height = box_tensor[3] - box_tensor[1]

            reduced_width = box_width * self.RF
            reduced_height = box_height * self.RF

            reduced_box = torch.tensor([
                box_center_x - reduced_width / 2,
                box_center_y - reduced_height / 2,
                box_center_x + reduced_width / 2,
                box_center_y + reduced_height / 2
            ], dtype=torch.float32).cuda()

            available_points = ~global_mask

            if not torch.any(available_points):
                break

            available_points_2d = points_2d[available_points]

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

            selected_points = points_2d[original_indices]
            masked_depths = pc_velo[original_indices, 0]

            color_indices = torch.minimum(
                torch.floor(510.0 / masked_depths).to(torch.int32),
                torch.tensor(255, dtype=torch.int32)
            )

            pts_2d = selected_points.cpu().numpy().astype(int)
            color_indices = color_indices.cpu().numpy()

            for (x, y), color_idx in zip(pts_2d, color_indices):
                cv2.circle(img, (x, y), 2, tuple(cmap[color_idx]), -1)

            if len(masked_depths) > 2:
                filtered_distances = self.filter_outliers(masked_depths.cpu().numpy())
                best_distance = self.get_best_distance(filtered_distances)

                text_x = int(box[0] + (box[2] - box[0]) / 2)
                text_y = int(box[1] + (box[3] - box[1]) / 2)
                cv2.putText(
                    img,
                    f"{best_distance:.2f}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 0, 0),
                    3,
                    cv2.LINE_AA
                )

    def pipeline(self, image, point_cloud, frame):

        lidar_fov_image = self.get_lidar_on_image_fov(point_cloud[:, :3], image)

        result, pred_bboxes = self.run_obstacle_detection(image, frame)

        if pred_bboxes.any():
            self.lidar_camera_fusion(pred_bboxes, result)

        return lidar_fov_image, result
