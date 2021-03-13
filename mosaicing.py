import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path


def get_frames(video_path, central_frame, mod):
    frames = []
    cap = cv2.VideoCapture(video_path)
    index = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if (index % mod) == 0:
            frames.append(frame[320:, :])
            # cv2.imshow('Basket', frame)

        if not ret or frame is None:
            cap.release()
            print("Released Video Resource")
            break

        if cv2.waitKey(20) == ord('q'): break
        index += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Number of frames : {len(frames)}")
    plt.title(f"Centrale {frames[central_frame].shape}")
    plt.imshow(frames[central_frame])
    plt.show()

    return frames


def collage(frames, direction=1):
    sift = cv2.xfeatures2d.SIFT_create()  # sift instance

    if direction == 1:
        current_mosaic = frames[0]
    else:
        current_mosaic = frames[-1]

    for i in range(len(frames) - 1):

        # FINDING FEATURES
        kp1 = sift.detect(current_mosaic)
        kp1, des1 = sift.compute(current_mosaic, kp1)
        kp2 = sift.detect(frames[i * direction + direction])
        kp2, des2 = sift.compute(frames[i * direction + direction], kp2)

        # MATCHING
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # Finding an homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        '''
        # Drawing matches
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(current_mosaic, kp1, frames[i * direction + direction],
                               kp2, good, None, **draw_params)
        plt.figure(figsize=(16, 8))
        plt.imshow(img3)
        plt.show()
        '''



        result = cv2.warpPerspective(frames[i * direction + direction],
                                     M,
                                     (current_mosaic.shape[1] + frames[i * direction + direction].shape[1],
                                      frames[i * direction + direction].shape[0] + 50))

        result[:current_mosaic.shape[0], :current_mosaic.shape[1]] = current_mosaic
        current_mosaic = result

        for j in range(len(current_mosaic[0])):
            if np.sum(current_mosaic[:, j]) == 0:
                current_mosaic = current_mosaic[:, :j - 50]
                break

        '''            
        # Displaying the result
        plt.figure(figsize=(16, 8))
        plt.title(current_mosaic.shape)
        plt.imshow(current_mosaic)
        plt.show()
        '''

    return current_mosaic


def add_frame(frame, pano):
    sift = cv2.xfeatures2d.SIFT_create()  # sift instance

    print('ciao')
    # FINDING FEATURES
    kp1 = sift.detect(pano)
    kp1, des1 = sift.compute(pano, kp1)
    kp2 = sift.detect(frame)
    kp2, des2 = sift.compute(frame, kp2)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print(f"Number of good correspondences: {len(good)}")

    # Finding an homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    result = cv2.warpPerspective(frame,
                                 M,
                                 (pano.shape[1],
                                  pano.shape[0]))

    avg_pano = np.where(result == 0, pano,
                        np.uint8(np.average(np.array([pano, result]), axis=0, weights=[1, 0.7])))

    plt.figure(figsize=(16, 8))
    plt.title(f"Warped new image: {pano.shape}")
    plt.imshow(result)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.title(f"AVG new image: {pano.shape}")
    plt.imshow(avg_pano)
    plt.tight_layout()
    plt.show()

    return avg_pano


if __name__ == '__main__':

    if os.path.exists('pano.png'):
        pano = cv2.imread("pano.png")
    else:
        central_frame = 36
        frames = get_frames('resources/Short4Mosaicing.mp4', central_frame, mod=3)

        frames_flipped = [cv2.flip(frames[i], 1) for i in range(central_frame)]

        current_mosaic1 = collage(frames[central_frame:], direction=1)
        current_mosaic2 = collage(frames_flipped, direction=-1)
        current_mosaic = collage([cv2.flip(current_mosaic2, 1), current_mosaic1])
        pano = current_mosaic
        cv2.imwrite("pano.png", pano)

    plt.figure(figsize=(16, 8))
    plt.title(f"Panorama: {pano.shape}")
    plt.imshow(pano)
    plt.tight_layout()
    plt.show()

    frame = cv2.imread("resources/snapshot.png")[320:]
    frame2 = cv2.imread("resources/snapshot2.png")[320:]
    frame3 = cv2.imread("resources/snapshot3.png")[320:]
    pano = add_frame(frame, pano)
    pano = add_frame(frame2, pano)
    pano = add_frame(frame3, pano)

    ### BINARIZATION, EROSION AND DILATION
    pano = cv2.cvtColor(pano, cv2.COLOR_RGB2GRAY)
    th, pano_otsu = cv2.threshold(pano, thresh=100, maxval=255, type=cv2.THRESH_OTSU)

    kernel = np.ones((20, 20), np.uint8)
    pano = cv2.erode(pano_otsu, kernel, iterations=1)
    pano = cv2.dilate(pano, kernel, iterations=1)

    plt.figure(figsize=(16, 8))
    plt.title(f"After Erosion-Dilation: {pano.shape}")
    plt.imshow(pano, cmap="gray")
    plt.tight_layout()
    plt.show()

    ### BLOB DETECTION
    params = cv2.SimpleBlobDetector_Params()
    # Filter by Area
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 1000000000000000

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = False
    params.minDistBetweenBlobs = 3
    params.minThreshold = 10
    params.maxThreshold = 200

    # adding a little frame
    pano[-4: -1] = pano[0:3] = 0
    pano[:, 0:3] = pano[:, -4:-1] = 0

    pano = 255 - pano
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(pano)
    print(keypoints)
    pano_with_blobs = cv2.drawKeypoints(pano, keypoints, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure(figsize=(16, 8))
    plt.title(f"After blob detection: {pano_with_blobs.shape}")
    plt.imshow(255 - pano_with_blobs, cmap="gray")
    plt.tight_layout()
    plt.show()

    ### BLOB FILTERING
