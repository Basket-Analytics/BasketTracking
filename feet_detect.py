import torch
import cv2
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from plot_tools import plt_plot

COLORS = {  # in HSV FORMAT
    'green': ([56, 50, 50], [100, 255, 255], [72, 200, 153]),  # NIGERIA
    'gray': ([0, 0, 0], [255, 35, 70], [120, 0, 0]),  # REFEREE
    'white': ([0, 0, 191], [255, 38, 255], [255, 0, 255])  # USA
}


class FeetDetector:

    def __init__(self, map2d):
        # Image segmentation model from DETECTRON2
        cfg_seg = get_cfg()
        cfg_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg_seg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor_seg = DefaultPredictor(cfg_seg)
        self.bbs = []
        self.map2d = map2d

    @staticmethod
    def count_non_black(image):
        colored = 0
        for color in image.flatten():
            if color > 0.0001:
                colored += 1
        return colored

    def get_players_pos(self, M, M1, frame):
        warped_kpts = []
        outputs_seg = self.predictor_seg(frame)

        indices = outputs_seg["instances"].pred_classes.cpu().numpy()
        predicted_masks = outputs_seg["instances"].pred_masks.cpu().numpy()

        ppl = []
        for i, entry in enumerate(indices):  # picking only class 0 (people)
            if entry == 0:
                ppl.append(predicted_masks[i])

        indexes_ppl = np.array(
            [np.array(np.where(p == True)).T for p in ppl])  # returns two np arrays per person, one for x one for y

        # calculate estimated position of players in the 2D map
        for keypoint, p in zip(indexes_ppl, ppl):

            top = min(keypoint[:, 0])
            bottom = max(keypoint[:, 0])
            left = min(keypoint[:, 1])
            right = max(keypoint[:, 1])

            tmp_tensor = p.reshape((p.shape[0], p.shape[1], 1))
            crop_img = np.where(tmp_tensor, frame, 0)

            crop_img = crop_img[top:bottom, left:right]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

            best_mask = [0, '']  # (num_non_black, color)
            for color in COLORS.keys():
                mask = cv2.inRange(crop_img, np.array(COLORS[color][0]), np.array(COLORS[color][1]))
                output = cv2.bitwise_and(crop_img, crop_img, mask=mask)

                non_blacks = FeetDetector.count_non_black(output)
                if best_mask[0] < non_blacks:
                    best_mask[0] = non_blacks
                    best_mask[1] = color

            head = int(np.argmin(keypoint[:, 0]))
            foot = int(np.argmax(keypoint[:, 0]))

            kpt = np.array([keypoint[head, 1], keypoint[foot, 0], 1])  # perspective space
            homo = M1 @ (M @ kpt.reshape((3, -1)))
            homo = np.int32(homo / homo[-1]).ravel()
            if best_mask[1] != '':
                color = np.array(cv2.cvtColor(np.uint8([[COLORS[best_mask[1]][2]]]), cv2.COLOR_HSV2BGR)).ravel()
                color = (int(color[0]), int(color[1]), int(color[2]))
                warped_kpts.append((homo, color))  # appending also the color
                cv2.circle(frame, (keypoint[head, 1], keypoint[foot, 0]), 2, color, 5)

        [cv2.circle(self.map2d, (k[0][0], k[0][1]), 10, (k[1]), 7) for k in warped_kpts]
        [cv2.circle(self.map2d, (k[0][0], k[0][1]), 13, (0, 0, 0), 3) for k in warped_kpts]  # adds border
        cv2.imshow("Tracking",
                   np.vstack((frame, cv2.resize(self.map2d, (frame.shape[1], frame.shape[1] // 2)))))
        return warped_kpts, frame


if __name__ == '__main__':
    print(torch.__version__, torch.cuda.is_available())
