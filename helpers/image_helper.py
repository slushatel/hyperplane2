import cv2
import numpy as np


def convert_image(workDir, pictureFilename, homography, image_size):
    orig = cv2.imread(workDir + '\snapshots\\' + pictureFilename, cv2.IMREAD_UNCHANGED)
    orig = cv2.cvtColor(orig, cv2.COLOR_RGB2RGBA)
    warp = cv2.warpPerspective(orig, homography, image_size)
    cv2.imwrite(workDir + '\warped\\' + pictureFilename, warp)
    return warp


def apply_homography(cam, u, v, workDir, pictureFilename, scale, z=0, plane=[], ax=0):
    if plane.__len__() == 0:
        pw_corners = cam.image_to_world(np.array([[0, 0], [u, 0], [0, v], [u, v]]).transpose(), z)
    else:
        pw_corners = cam.image_to_world_plane(np.array([[0, 0], [u, 0], [0, v], [u, v]]).transpose(), plane, ax)

    if ax != 0:
        ax.scatter(pw_corners[0], pw_corners[1], pw_corners[2], color='blue')

    points1 = np.array([[0, 0], [u, 0], [0, v], [u, v]])

    # translate into positive space
    trans = np.hstack((np.eye(3, 2), [[-pw_corners[0].min()], [-pw_corners[1].min()], [1]]))
    pw_corners_0 = np.vstack((pw_corners[0:2], np.ones([1, 4])))
    scale = np.diag([scale, scale, 1])
    points2 = scale.dot(trans).dot(pw_corners_0)[0:2].transpose()[::-1]

    homography = cv2.getPerspectiveTransform(points1.astype(np.float32), points2.astype(np.float32))

    size_array = np.amax(points2, axis=0).astype(int)
    warp = convert_image(workDir, pictureFilename, homography, (size_array[0], size_array[1]))

    img_square = np.vstack((
        np.array([[pw_corners[0].min(), pw_corners[1].max()], [pw_corners[0].max(), pw_corners[1].max()],
                  [pw_corners[0].min(), pw_corners[1].min()], [pw_corners[0].max(), pw_corners[1].min()]
                  ]).transpose(),
        np.ones([1, 4])))
    return warp, scale.dot(img_square)
