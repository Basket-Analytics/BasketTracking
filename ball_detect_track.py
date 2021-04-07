import cv2
import os.path
import numpy as np

from plot_tools import plt_plot
from feet_detection import get_players_pos
from main import TOPCUT

CHECK_TRACK = 5
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

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

def ball_detection(img_train_dir, query_frame, th = 0.98):
    img_train = []
    for file in os.listdir(img_train_dir):
        img_train.append(cv2.imread(img_train_dir + file, 0))

    img_query = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)

    centers = circle_detect(img_query)
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
                    res = cv2.matchTemplate(focus, ball, cv2.TM_CCORR_NORMED)  # NORMALIZED CROSS CORRELATION
                    if np.max(res) >= th:
                        return bb

    return None


def find_ball_video(video, map2d):
    while video.isOpened():
        ok, frame = video.read()
        if ok:
            frame = frame[TOPCUT:, :]
            # if resized_map is not None: frame[-(resized_map.shape[0]):, -(resized_map.shape[1]):] = resized_map
            cv2.imshow("Tracking", np.vstack((frame, cv2.resize(map2d, (frame.shape[1], frame.shape[1] // 2)))))
            bb = ball_detection('resources/ball/', frame)
            if bb is not None:
                break
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            return None, None

    frame = cv2.rectangle(frame, (bb[0][0], bb[0][1]), (bb[1][0], bb[1][1]), (0, 255, 0), 4)
    # if resized_map is not None: frame[-(resized_map.shape[0]):, -(resized_map.shape[1]):] = resized_map
    cv2.imshow("Tracking", np.vstack((frame, cv2.resize(map2d, (frame.shape[1], frame.shape[1] // 2)))))
    return bb, frame

def get_homography(frame, sift, des1, kp1):
    kp2 = sift.detect(frame)
    kp2, des2 = sift.compute(frame, kp2)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return M


def ball_tracker(video_directory, map2d, predictor):
    global resized_map
    resized_map = None

    M1 = np.load("Rectify1.npy")

    def init_tracker(video):
        # Initialize tracker with first frame and bounding box
        bbox, frame = find_ball_video(video, map2d)
        if bbox is not None:
            bbox = (bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])
            tracker.init(frame, bbox)
        return bbox, frame

    tracker_types = 'CSRT'
    tracker = cv2.TrackerCSRT_create()

    video = cv2.VideoCapture("resources/Short4Mosaicing.mp4")
    bbox, frame = init_tracker(video)

    found = True
    check_tracking = CHECK_TRACK

    # sift init
    sift = cv2.xfeatures2d.SIFT_create()
    pano = cv2.imread("pano_enhanced.png")
    kp1 = sift.detect(pano)
    kp1, des1 = sift.compute(pano, kp1)

    padding = 30

    while video.isOpened():

        ok, frame = video.read()
        if ok:
            frame = frame[TOPCUT:, :]
            # Update tracker
            if found: res, bbox = tracker.update(frame)

            # Draw bounding box
            if check_tracking > 0:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                ball_center = np.array([int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2), 1])

                clean_frame = frame.copy()
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                found = True

                # DRAW POINT/BALL IN THE MAP
                M = get_homography(frame, sift, des1, kp1)
                homo = M1 @ (M @ ball_center.reshape((3, -1)))
                homo = np.int32(homo / homo[-1]).ravel()

                warped_kpts, frame = get_players_pos(frame, M, M1, predictor)
                [cv2.circle(map2d, (k[0][0], k[0][1]), 10, (k[1]), 7) for k in warped_kpts]
                [cv2.circle(map2d, (k[0][0], k[0][1]), 13, (0, 0, 0), 3) for k in warped_kpts]  # ads border

                cv2.circle(map2d, (homo[0], homo[1]), 10, (0, 0, 255), 5)  # for the ball
                # bottom right map visualization
                # resized_map = cv2.resize(map2d, (400, 200))
                # frame[-(resized_map.shape[0]):, -(resized_map.shape[1]):] = resized_map

                cv2.imshow("Tracking", np.vstack((frame, cv2.resize(map2d, (frame.shape[1], frame.shape[1] // 2)))))

            elif ball_detection('resources/ball/',
                                clean_frame[p1[1]-padding:p2[1]+padding, p1[0]-padding:p2[0]+padding],
                                0.8) is not None:
                print("entro")
                check_tracking = CHECK_TRACK

            else:
                check_tracking = CHECK_TRACK
                bbox, frame = init_tracker(video)
                if bbox is None:
                    break

            cv2.putText(frame, tracker_types + " Tracker",
                        (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            k = cv2.waitKey(5) & 0xff
            if k == 27: break

            check_tracking -= 1

        else:
            break
