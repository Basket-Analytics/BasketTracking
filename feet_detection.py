import torch
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

if __name__ == '__main__':
    print(torch.__version__, torch.cuda.is_available())

    video = cv2.VideoCapture("resources/Short4Mosaicing.mp4")

    while video.isOpened():
        ok, frame = video.read()
        if ok:
            frame = frame[320:, :]

            # Inference with a keypoint detection model
            cfg = get_cfg()  # get a fresh new config
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
            predictor = DefaultPredictor(cfg)
            outputs = predictor(frame)
            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow("image", out.get_image()[:, :, ::-1])

        k = cv2.waitKey(5) & 0xff
        if k == 27: break
