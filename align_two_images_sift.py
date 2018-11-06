from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

workDir = 'c:\work\windpropeller'

MIN_MATCH_COUNT = 10


def alignImages(pictureNumber, im1, im2):
    # img1 = cv2.imread('box.png', 0)  # queryImage
    # img2 = cv2.imread('box_in_scene.png', 0)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = im1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        im2 = cv2.polylines(im2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print
        "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    imMatches = cv2.drawMatches(im1, kp1, im2, kp2, good, None, **draw_params)
    cv2.imwrite(workDir + "/matches_2_" + str(pictureNumber) + "-" + str(pictureNumber + 1) + ".jpg", imMatches)


#####################################

if __name__ == '__main__':

    pictureNumbersMin = 76
    pictureNumbersMax = 98
    i = 0
    n = pictureNumbersMax - pictureNumbersMin + 1
    for i in range(0, n - 1):
        pictureNumber = pictureNumbersMin + i
        imReference = cv2.imread(workDir + '\snapshots\snapshot000' + str(pictureNumber) + '.jpg', cv2.IMREAD_UNCHANGED)
        im = cv2.imread(workDir + '\snapshots\snapshot000' + str(pictureNumber + 1) + '.jpg', cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        alignImages(pictureNumber, im, imReference)

    # Read reference image
    refFilename = "form.jpg"
    print("Reading reference image : ", refFilename)
    # imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    imReference = cv2.imread(workDir + '\snapshots\snapshot000' + str(98) + '.jpg', cv2.IMREAD_UNCHANGED)

    # Read image to be aligned
    imFilename = "scanned-form.jpg"
    print("Reading image to align : ", imFilename);
    # im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    im = cv2.imread(workDir + '\snapshots\snapshot000' + str(97) + '.jpg', cv2.IMREAD_UNCHANGED)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    alignImages(97, im, imReference)

    # Write aligned image to disk.
    outFilename = workDir + "/aligned.jpg"
    print("Saving aligned image : ", outFilename);
    # cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    # print("Estimated homography : \n", h)
