import cv2
import os.path
import numpy as np

from plot_tools import plt_plot
from main import TOPCUT

MAX_TRACK = 20
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


def ball_detection(img_train_dir, query_frame):
    img_train = []
    for file in os.listdir(img_train_dir):
        img_train.append(cv2.imread(img_train_dir + file, 0))

    img_query = cv2.cvtColor(query_frame, cv2.COLOR_RGB2GRAY)

    centers = circle_detect(img_query)
    if centers is not None:
        af = 7
        bbs = [([c[0] - c[2] - af, c[1] - c[2] - af],  # tl
                # [c[0] + c[2] + 5, c[1] - c[2] - 5], #tr
                [c[0] + c[2] + af, c[1] + c[2] + af])  # br
               # [c[0] - c[2] - 5, c[1] + c[2] + 5]) #bl
               for c in centers]

        # Creating SIFT object
        # sift = cv2.xfeatures2d.SIFT_create()

        for ball in img_train:
            # kp_train = sift.detect(ball)
            # kp_train, des_train = sift.compute(ball, kp_train)
            for bb in bbs:
                focus = img_query[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]]
                if focus.shape[0] > ball.shape[0] and focus.shape[1] > ball.shape[1]:
                    res = cv2.matchTemplate(focus, ball, cv2.TM_CCORR_NORMED)  # NORMALIZED CROSS CORRELATION
                    if np.max(res) >= 0.98:
                        return bb
                """ focus = img_query[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]]

                # Detecting Keypoints in the two images
                kp_query = sift.detect(focus)

                # Computing the descriptors for each keypoint
                kp_query, des_query = sift.compute(focus, kp_query)

                # Initializing the matching algorithm
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                # Matching the descriptors
                matches = flann.knnMatch(des_query, des_train, k=2)


                # Keeping only good matches as per Lowe's ratio test.
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
                if len(good) >= 4:
                    img_query = cv2.rectangle(img_query, (bb[0][0], bb[0][1]), (bb[1][0], bb[1][1]), (0, 255, 0), 4)
                    plt_plot(img_query, cmap='gray')
                    return bb
                    """

    return None


def find_ball_video(video):
    pano = cv2.imread("pano_enhanced.png")
    while video.isOpened():
        ok, frame = video.read()
        if ok:
            frame = frame[TOPCUT:, :]
            resized = cv2.resize(pano, (frame.shape[1], frame.shape[0]))
            cv2.imshow("Tracking", np.vstack((frame, resized)))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            bb = ball_detection('resources/ball/', frame)
            if bb is not None:
                break
        else:
            return None, None

    img_query = cv2.rectangle(frame, (bb[0][0], bb[0][1]), (bb[1][0], bb[1][1]), (0, 255, 0), 4)
    cv2.imshow("Tracking", np.vstack((img_query, resized)))
    return bb, frame


def ball_tracker(video_directory):
    def init_tracker(video):
        # Initialize tracker with first frame and bounding box
        bbox, frame = find_ball_video(video)
        if bbox is not None:
            bbox = (bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1])
            tracker.init(frame, bbox)
        return bbox, frame

    tracker_types = 'CSRT'
    tracker = cv2.TrackerCSRT_create()

    video = cv2.VideoCapture("resources/Short4Mosaicing.mp4")
    bbox, frame = init_tracker(video)

    found = True
    max_tracking = MAX_TRACK

    # sift init
    sift = cv2.xfeatures2d.SIFT_create()
    pano = cv2.imread("pano_enhanced.png")
    kp1 = sift.detect(pano)
    kp1, des1 = sift.compute(pano, kp1)

    while video.isOpened():
        # Read a new frame
        ok, frame = video.read()
        if ok:
            frame = frame[TOPCUT:, :]
            # Update tracker
            if found: res, bbox = tracker.update(frame)

            # Draw bounding box
            if max_tracking > 0:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                found = True
            else:
                # after 10 frames detect
                max_tracking = MAX_TRACK
                bbox, frame = init_tracker(video)
                if bbox is None:
                    break

            # Display tracker type on frame
            cv2.putText(frame, tracker_types + " Tracker",
                        (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display result
            # cv2.imshow("Tracking", frame)
            # Exit if ESC pressed
            k = cv2.waitKey(5) & 0xff
            if k == 27: break

            kp2 = sift.detect(frame)
            kp2, des2 = sift.compute(frame, kp2)

            matches = flann.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            # Finding an homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            result = cv2.warpPerspective(frame,
                                         M,
                                         (pano.shape[1],
                                          pano.shape[0]))

            avg_pano = np.uint8(np.average(np.array([pano, result]), axis=0, weights=[1, 0.7]))
            resized = cv2.resize(avg_pano, (frame.shape[1], frame.shape[0]))
            cv2.imshow("Tracking", np.vstack((frame, resized)))

            max_tracking -= 1


        else:
            break
