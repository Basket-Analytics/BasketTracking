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
    keypoints = outputs["instances"].pred_keypoints
    keypoints = keypoints.cpu().numpy()

    for keypoint in keypoints:
        head = int(np.argmin(keypoint[:, 1]))
        foot = int(np.argmax(keypoint[:, 1]))
        kpt = np.array([keypoint[head, 0], keypoint[foot, 1], 1])  # perspetcive space
        homo = M1 @ (M @ kpt.reshape((3, -1)))
        homo = np.int32(homo / homo[-1]).ravel()
        warped_kpts.append(homo)

        cv2.circle(frame, (keypoint[head, 0], keypoint[foot, 1]), 2, (0, 255, 0), 5)

    # draws skeletons
    '''v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))'''

    return warped_kpts, frame


if __name__ == '__main__':
    print(torch.__version__, torch.cuda.is_available())
