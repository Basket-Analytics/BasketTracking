import cv2
import os.path
import numpy as np

from plot_tools import plt_plot
from feet_detect import *
from main import TOPCUT

MAX_TRACK = 5
IOU_BALL_PADDING = 30


class BallDetectTrack:
    def __init__(self, players):
        self.ball_padding = 30
        self.check_track = MAX_TRACK
        self.do_detection = True
        self.tracker_type = 'CSRT'
        self.tracker = cv2.TrackerCSRT_create()
        self.players = players

    @staticmethod
    def circle_detect(img, plot=False):
        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=25, minRadius=5, maxRadius=15)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            if plot:
                for i in circles[0, :]:
                    # draw the outer circle and center of the circle
                    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

                plt_plot(cimg, "Detected Circles")

            return circles.reshape((-1, 3))

    def ball_detection(self, img_train_dir, query_frame, th=0.98):
        img_train = []
        for file in os.listdir(img_train_dir):
            img_train.append(cv2.imread(img_train_dir + file, 0))

        img_query = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)

        centers = self.circle_detect(img_query)
        if centers is not None:
            af = 7
            bbs = [([c[0] - c[2] - af, c[1] - c[2] - af],  # tl
                    # [c[0] + c[2] + 5, c[1] - c[2] - 5], #tr
                    [c[0] + c[2] + af, c[1] + c[2] + af])  # br
                   # [c[0] - c[2] - 5, c[1] + c[2] + 5]) #bl
                   for c in centers]

            for ball in img_train:
                for bb in bbs:
                    focus = img_query[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]]
                    if focus.shape[0] > ball.shape[0] and focus.shape[1] > ball.shape[1]:
                        # NORMALIZED CROSS CORRELATION
                        res = cv2.matchTemplate(focus, ball, cv2.TM_CCORR_NORMED)
                        if np.max(res) >= th:
                            bbox = bb
                            bb = (bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])
                            return bb
        return None

    def ball_tracker(self, M, M1, frame, map_2d, map_2d_text, timestamp):

        if self.do_detection:
            bbox = self.ball_detection("resources/ball/", frame)
            if bbox is not None:
                self.tracker.init(frame, bbox)
                self.do_detection = not self.do_detection
        else:
            res, bbox = self.tracker.update(frame)

        if bbox is not None:

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            ball_center = np.array([int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2), 1])
            clean_frame = frame.copy()

            bbox_iou = (ball_center[1] - IOU_BALL_PADDING,
                        ball_center[0] - IOU_BALL_PADDING,
                        ball_center[1] + IOU_BALL_PADDING,
                        ball_center[0] + IOU_BALL_PADDING)
            scores = []

            for p in self.players:
                try:
                    tmp = p.positions[timestamp]
                    if p.team != "referee":
                        if p.previous_bb is not None:
                            scores.append((p, FeetDetector.bb_intersection_over_union(bbox_iou, p.previous_bb)))
                except KeyError:
                    pass

            if len(scores) > 0:
                for p in self.players:
                    p.has_ball = False
                max_score = max(scores, key=itemgetter(1))
                max_score[0].has_ball = True
                cv2.circle(map_2d_text, (max_score[0].positions[timestamp]), 27, (0, 0, 255), 10)

            if self.check_track > 0:
                homo = M1 @ (M @ ball_center.reshape((3, -1)))
                homo = np.int32(homo / homo[-1]).ravel()
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.circle(map_2d, (homo[0], homo[1]), 10, (0, 0, 255), 5)  # for the ball on the 2D map
                self.check_track -= 1

            elif self.ball_detection('resources/ball/',
                                     clean_frame[p1[1] - self.ball_padding:p2[1] + self.ball_padding,
                                     p1[0] - self.ball_padding:p2[0] + self.ball_padding],
                                     0.5) is not None:
                self.check_track = MAX_TRACK
                self.do_detection = False

            else:  # se è 0 check track e non ho trovato la ball
                self.check_track = MAX_TRACK
                self.do_detection = True

            return frame, map_2d

        else:
            return frame, None
