from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

workDir = 'c:\work\windpropeller'


def alignImages(pictureNumber, im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite(workDir + "/matches" +str(pictureNumber) + "-" + str(pictureNumber+1)+ ".jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    # im1Reg = cv2.warpPerspective(im1, h, (width, height))

    # return im1Reg, h

# img = cv2.imread(workDir + '\snapshots\snapshot000' + str(pictureNumber) + '.jpg', cv2.IMREAD_UNCHANGED)

if __name__ == '__main__':

    pictureNumbersMin = 76
    pictureNumbersMax = 98
    i = 0
    n = pictureNumbersMax - pictureNumbersMin + 1
    for i in range(0, n-1):
        pictureNumber = pictureNumbersMin + i
        imReference = cv2.imread(workDir + '\snapshots\snapshot000' + str(pictureNumber) + '.jpg', cv2.IMREAD_UNCHANGED)
        im = cv2.imread(workDir + '\snapshots\snapshot000' + str(pictureNumber+1) + '.jpg', cv2.IMREAD_UNCHANGED)
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
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = workDir + "/aligned.jpg"
    print("Saving aligned image : ", outFilename);
    # cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)