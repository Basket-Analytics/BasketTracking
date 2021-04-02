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

COLORS = {  # in BGR FORMAT
    'green': ([17, 75, 15], [50, 250, 56]),  # green
    'blue': ([43, 15, 4], [250, 88, 50]),  # blue
    'white': ([187, 169, 112], [255, 255, 255])  # white
}


def count_non_black(image):
    colored = 0
    for color in image.flatten():
        if color > 0.0001:
            colored += 1
    return colored


def get_players_pos(frame, M, M1):
    warped_kpts = []

    # Inference with a keypoint detection model
    cfg = get_cfg()  # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))  # LOAD MODEL
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # LOAD WEIGHTS
    predictor = DefaultPredictor(cfg)
    outputs = predictor(frame)

    '''# Image segmentation model
    cfg_seg = get_cfg()
    cfg_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg_seg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor_seg = DefaultPredictor(cfg_seg)
    outputs_seg = predictor(frame)'''

    bbs = outputs['instances'].pred_boxes.tensor.cpu().numpy()

    # cv2.rectangle(frame, (bbs[0][0], bbs[0][1]), (bbs[0][2], bbs[0][3]), (0, 255, 0), 4)

    keypoints = outputs["instances"].pred_keypoints
    keypoints = keypoints.cpu().numpy()

    # calculate estimated position of players in the 2D map
    for keypoint, bb in zip(keypoints, bbs):

        bb = np.int32(bb)
        crop_img = frame[bb[1]:bb[3], bb[0]:bb[2]]
        best_mask = [0, '']  # (num_non_black, color)
        for color in COLORS.keys():
            mask = cv2.inRange(crop_img, np.array(COLORS[color][0]), np.array(COLORS[color][1]))
            output = cv2.bitwise_and(crop_img, crop_img, mask=mask)

            non_blacks = count_non_black(output)
            if best_mask[0] < non_blacks:
                best_mask[0] = non_blacks
                best_mask[1] = color

        head = int(np.argmin(keypoint[:, 1]))
        foot = int(np.argmax(keypoint[:, 1]))
        kpt = np.array([keypoint[head, 0], keypoint[foot, 1], 1])  # perspective space
        homo = M1 @ (M @ kpt.reshape((3, -1)))
        homo = np.int32(homo / homo[-1]).ravel()
        warped_kpts.append((homo, COLORS[best_mask[1]][1]))  # appending also the color

        cv2.circle(frame, (keypoint[head, 0], keypoint[foot, 1]), 2, (COLORS[best_mask[1]][1]), 5)

        # draws skeletons
        '''v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("frame", out.get_image()[:, :, ::-1])'''

    return warped_kpts, frame


if __name__ == '__main__':
    print(torch.__version__, torch.cuda.is_available())
