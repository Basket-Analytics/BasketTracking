import cv2
import numpy as np

from plot_tools import plt_plot


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

        result = cv2.warpPerspective(frames[i * direction + direction],
                                     M,
                                     (current_mosaic.shape[1] + frames[i * direction + direction].shape[1],
                                      frames[i * direction + direction].shape[0] + 50))

        result[:current_mosaic.shape[0], :current_mosaic.shape[1]] = current_mosaic
        current_mosaic = result

        # removing black part of the collage
        for j in range(len(current_mosaic[0])):
            if np.sum(current_mosaic[:, j]) == 0:
                current_mosaic = current_mosaic[:, :j - 50]
                break

    return current_mosaic


def add_frame(frame, pano, pano_enhanced, plot=False):
    sift = cv2.xfeatures2d.SIFT_create()  # sift instance

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
    if len(good) < 70: return pano

    # Finding an homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    result = cv2.warpPerspective(frame,
                                 M,
                                 (pano.shape[1],
                                  pano.shape[0]))

    if plot: plt_plot(result, "Warped new image")

    avg_pano = np.where(result < 100, pano_enhanced,
                        np.uint8(np.average(np.array([pano_enhanced, result]), axis=0, weights=[1, 0.7])))
    # fare la mediana, dare 3 CFU a Simone

    if plot: plt_plot(avg_pano, "AVG new image")

    return avg_pano


def binarize_erode_dilate(img, plot=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, img_otsu = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_OTSU)

    if plot: plt_plot(img_otsu, "Panorama after Otsu", cmap="gray")

    kernel = np.array([[0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0]], np.uint8)

    img_otsu = cv2.erode(img_otsu, kernel, iterations=20)
    img_otsu = cv2.dilate(img_otsu, kernel, iterations=20)

    if plot: plt_plot(img_otsu, "After Erosion-Dilation", cmap="gray")
    return img_otsu


def rectangularize_court(pano, plot=False):
    # BLOB FILTERING & BLOB DETECTION

    # adding a little frame to enable detection
    # of blobs that touch the borders
    pano[-4: -1] = pano[0:3] = 0
    pano[:, 0:3] = pano[:, -4:-1] = 0

    mask = np.zeros(pano.shape, dtype=np.uint8)
    cnts = cv2.findContours(pano, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_court = []

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    threshold_area = 100000
    for c in cnts:
        area = cv2.contourArea(c)
        if area > threshold_area:
            cv2.drawContours(mask, [c], -1, (36, 255, 12), -1)
            contours_court.append(c)

    pano = mask
    if plot: plt_plot(pano, "After Blob Detection", cmap="gray")

    # pano = 255 - pano
    contours_court = contours_court[0]
    simple_court = np.zeros(pano.shape)

    # convex hull
    hull = cv2.convexHull(contours_court)
    cv2.drawContours(pano, [hull], 0, 100, 2)
    if plot: plt_plot(pano, "After ConvexHull", cmap="gray",
                      additional_points=hull.reshape((-1, 2)))

    # fitting a poly to the hull
    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    corners = approx.reshape(-1, 2)
    cv2.drawContours(pano, [approx], 0, 100, 5)
    cv2.drawContours(simple_court, [approx], 0, 255, 3)

    if plot:
        plt_plot(pano, "After Rectangular Fitting", cmap="gray")
        plt_plot(simple_court, "Rectangularized Court", cmap="gray")
        print("simplified contour has", len(approx), "points")

    return simple_court, corners


def homography(rect, image, plot=False):
    bl, tl, tr, br = rect
    rect = np.array([tl, tr, br, bl], dtype="float32")

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB)) + 700

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    if plot: plt_plot(warped)
    return warped


def rectify(pano_enhanced, corners, plot=False):
    # TODO: adapt this in a way that works in any setting
    panoL = pano_enhanced[:, :1870]
    panoR = pano_enhanced[:, 1870:]
    cornersL = np.array([corners[0], corners[1], [1865, 55], [1869, 389]])
    cornersR = np.array(
        [[0, 389],
         [0, 55],
         [corners[2][0] - 1870, corners[2][1]],
         [corners[3][0] - 1870, corners[3][1]]
         ])
    homography(corners, pano_enhanced)
    h1 = homography(cornersL, panoL)
    h2 = homography(cornersR, panoR)
    # rectified = np.hstack((h1, cv2.resize(h2, (int((h2.shape[0] / h1.shape[0]) * h1.shape[1]), h1.shape[0]))))
    rectified = np.hstack((h1, cv2.resize(h2, (h1.shape[1], h1.shape[0]))))
    if plot: plt_plot(rectified)
    return rectified
