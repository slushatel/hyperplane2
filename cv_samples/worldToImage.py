from math import sqrt

import cv_samples.camera as camera
import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cv2
import helpers.plot_helper as plot_helper

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
ax = plot_helper.prepare_plot()


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


axis = [1, 0, 0]
theta = math.pi * 7 / 8

# print(np.dot(rotation_matrix(axis, theta), v))
######################################

c = camera.Camera()

u, v = 5456, 3632
camera_half_angle = math.pi * 38.2 / 180
f = u / 2 / math.tan(camera_half_angle)

c.set_K_elements(u / 2, v / 2, f=f)
# R = np.array(
#     [[1, 0, 0],
#      [0, 1, 0],
#      [0, 0, 1]])
R = np.array([[9.9730e-01, -7.2900e-02, 1.2000e-03],
              [-4.8200e-02, -6.4730e-01, 7.6070e-01],
              [-5.4700e-02, -7.5870e-01, -6.4910e-01]])
translationMatrix = np.array([[2.1872],
                              [-7.2424],
                              [6.5368]])
# R = np.array([[0.211, -0.306, -0.928],
#               [0.662, 0.742, -0.0947],
#               [0.718, -0.595, 0.360]])
# translationMatrix = np.array([[0.789],
#               [0.147],
#               [3.26]])
# R = np.dot(rotation_matrix(axis, theta), R)

# for i in range(0, 3):
#     len1 = sqrt(pow(R[0, i], 2) + pow(R[1, i], 2) + pow(R[2, i], 2))
#     print("The length of all rotation vectors: " + str(len1))

c.set_R(R)
# c.set_R_euler_angles([0, 0, 0])
# R = c.R

# cameraPosition = np.array([[0], [0], [50]])
# translationMatrix = -R.dot(cameraPosition)
c.set_t(translationMatrix)
cameraPosition = -R.transpose().dot(translationMatrix)
print("camera center: " + str(cameraPosition))

startPoint = np.array([[5., 10., 0.]])
point = c.world_to_image(startPoint.T)
print("point's image coordinates: " + str(point))

# camera point from opencv
imagePoints, jacobian = cv2.projectPoints(startPoint, R, translationMatrix, c.K, np.array([]))
print("point's image coordinates (using projectPoints): " + str(imagePoints))

#  world coordinates
pw = c.image_to_world_y(point, 0)
# needs tuning
# pw_alternative = c.R.transpose().dot(point) - c.R.transpose().dot(c.t)
pw_corners = c.image_to_world_y(np.array([[0, 0], [u, 0], [0, v], [u, v]]).transpose(), 0)

# Find homography
# points1 = np.zeros((4, 2), dtype=np.float32)
# points2 = np.zeros((4, 2), dtype=np.float32)
points1 = np.array([[0, 0], [u, 0], [0, v], [u, v]])
# points2 = pw_corners[0:2].transpose()*100

trans = np.hstack((np.eye(3, 2), [[-pw_corners[0].min()], [-pw_corners[1].min()], [1]]))
pw_corners_0 = np.vstack((pw_corners[0:2], np.ones([1, 4])))
sn = 10
scale = np.diag([sn, sn, 1])
points2 = scale.dot(trans).dot(pw_corners_0)[0:2].transpose()[::-1]

# h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
h = cv2.getPerspectiveTransform(points1.astype(np.float32), points2.astype(np.float32))

workDir = 'd:\work\windpropeller'
pictureNumber = 76
# Use homography
# height, width, channels = im2.shape
orig = cv2.imread(workDir + '\snapshots\snapshot000' + str(pictureNumber) + '.jpg')
size_array = np.amax(points2, axis=0).astype(int)
warp = cv2.warpPerspective(orig, h, (size_array[0], size_array[1]))
cv2.imwrite(workDir + '\warped\snapshot000' + str(pictureNumber) + '.jpg', warp)

print("image to world coordinates (z = 0):" + str(pw))
plot_helper.plot_point(ax, pw, 'green')

R *= 1

plot_helper.plot_vector(ax, cameraPosition[:, 0], R[0], colors['green'])  # green
plot_helper.plot_vector(ax, cameraPosition[:, 0], R[1], colors['red'])  # red
plot_helper.plot_vector(ax, cameraPosition[:, 0], R[2], colors['blue'])  # violet

# plot_helper.plot_plane(ax, cameraPosition, R[0], colors['green'])
# plot_helper.plot_plane(ax, cameraPosition, R[1], colors['blue'])

plot_helper.plot_point(ax, [0, 0, 0], 'red')
plot_helper.plot_point(ax, cameraPosition[:, 0], 'red')

plot_helper.set_axes_equal(ax)

plt.show()
