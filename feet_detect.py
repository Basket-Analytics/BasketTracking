import torch
import cv2
import numpy as np
from player import *
from operator import itemgetter

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from plot_tools import plt_plot

COLORS = {  # in HSV FORMAT
    'green': ([56, 50, 50], [100, 255, 255], [72, 200, 153]),  # NIGERIA
    'referee': ([0, 0, 0], [255, 35, 65], [120, 0, 0]),  # REFEREE
    'white': ([0, 0, 220], [255, 15, 255], [255, 0, 255])  # USA
}

IOU_TH = 0.3


def hsv2bgr(color_hsv):
    color_bgr = np.array(cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR)).ravel()
    color_bgr = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
    return color_bgr


class FeetDetector:

    def __init__(self):
        # Image segmentation model from DETECTRON2
        cfg_seg = get_cfg()
        cfg_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg_seg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor_seg = DefaultPredictor(cfg_seg)
        self.bbs = []
        self.players = []

        for i in range(1, 6):
            self.players.append(Player(i, 'green', hsv2bgr(COLORS['green'][2])))
            self.players.append(Player(i, 'white', hsv2bgr(COLORS['white'][2])))
        self.players.append(Player(0, 'referee', hsv2bgr(COLORS['referee'][2])))

    @staticmethod
    def count_non_black(image):
        colored = 0
        for color in image.flatten():
            if color > 0.0001:
                colored += 1
        return colored

    def bb_intersection_over_union(self, boxA, boxB):
        # sources: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])  # horizontal tl
        yA = max(boxA[1], boxB[1])  # vertical tl
        xB = min(boxA[2], boxB[2])  # horizontal br
        yB = min(boxA[3], boxB[3])  # vertical br
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou


    def get_players_pos(self, M, M1, frame, timestamp, map_2d):
        warped_kpts = []
        outputs_seg = self.predictor_seg(frame)

        indices = outputs_seg["instances"].pred_classes.cpu().numpy()
        predicted_masks = outputs_seg["instances"].pred_masks.cpu().numpy()

        ppl = []
        for i, entry in enumerate(indices):  # picking only class 0 (people)
            if entry == 0:
                ppl.append(predicted_masks[i])

        indexes_ppl = np.array(
            [np.array(np.where(p == True)).T for p in ppl])
        # returns two np arrays per person, one for x one for y

        # calculate estimated position of players in the 2D map
        for keypoint, p in zip(indexes_ppl, ppl):

            top = min(keypoint[:, 0])
            bottom = max(keypoint[:, 0])
            left = min(keypoint[:, 1])
            right = max(keypoint[:, 1])
            bbox_person = (top, left, bottom, right)

            tmp_tensor = p.reshape((p.shape[0], p.shape[1], 1))
            crop_img = np.where(tmp_tensor, frame, 0)

            crop_img = crop_img[top:(bottom - int(0.3 * (bottom - top))), left:right]
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
            # homo = [vertical pos, horizontal pos]
            # homo has the position of player in the 2D map

            if best_mask[1] != '':
                color = hsv2bgr(COLORS[best_mask[1]][2])
                warped_kpts.append((homo, color, best_mask[1], bbox_person))  # appending also the color
                cv2.circle(frame, (keypoint[head, 1], keypoint[foot, 0]), 2, color, 5)

        for kpt in warped_kpts:
            (homo, color, color_key, bbox) = kpt
            # updates if possible the player position and bbox
            iou_scores = []  # (current_iou, player)
            for player in self.players:
                if (player.team == color_key) and (player.previous_bb is not None) and \
                        (0 <= homo[0] < map_2d.shape[0]) and (0 <= homo[1] < map_2d.shape[1]):
                    iou_current = self.bb_intersection_over_union(bbox, player.previous_bb)
                    # print(iou_current)
                    if iou_current >= IOU_TH:
                        iou_scores.append((iou_current, player))

            if len(iou_scores) > 0:
                # only update player
                max_iou = max(iou_scores, key=itemgetter(0))
                max_iou[1].previous_bb = bbox
                max_iou[1].positions[timestamp] = (homo[0], homo[1])
            else:
                for player in self.players:
                    if (player.team == color_key) and (player.previous_bb is None):
                        player.previous_bb = bbox
                        player.positions[timestamp] = (homo[0], homo[1])
                        break

        for player in self.players:
            if len(player.positions) > 0:
                if (timestamp - max(player.positions.keys())) >= 5:
                    player.positions = {}
                    player.previous_bb = None

        map_2d_text = map_2d.copy()
        for p in self.players:
            if p.team != 'referee':
                try:
                    cv2.circle(map_2d, (p.positions[timestamp]), 10, p.color, 7)
                    cv2.circle(map_2d, (p.positions[timestamp]), 13, (0, 0, 0), 3)
                    cv2.circle(map_2d_text, (p.positions[timestamp]), 10, p.color, 7)
                    cv2.circle(map_2d_text, (p.positions[timestamp]), 13, (0, 0, 0), 3)
                    cv2.putText(map_2d_text, str(p.ID), (p.positions[timestamp]),
                                cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 0, 0), 2, cv2.LINE_AA)
                except KeyError:
                    pass

        return frame, map_2d, map_2d_text


if __name__ == '__main__':
    print(torch.__version__, torch.cuda.is_available())
