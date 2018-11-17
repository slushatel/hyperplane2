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
R = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]])

# R = np.dot(rotation_matrix(axis, theta), R)

for i in range(0, 3):
    len1 = sqrt(pow(R[0, i], 2) + pow(R[1, i], 2) + pow(R[2, i], 2))
    print("The length of all rotation vectors: " + str(len1))

# c.set_R(R)
c.set_R_euler_angles([0, 0, 0])
R = c.R

cameraPosition = np.array([[0], [0], [50]])
translationMatrix = -R.dot(cameraPosition)
c.set_t(translationMatrix)

print("camera center: " + str(-R.transpose().dot(translationMatrix)))

startPoint = np.array([[5., 10., 0.]])
point = c.world_to_image(startPoint.T)
print("point's image coordinates: " + str(point))

# camera point from opencv
imagePoints, jacobian = cv2.projectPoints(startPoint, R, translationMatrix, c.K, np.array([]))
print("point's image coordinates (using projectPoints): " + str(imagePoints))

#  world coordinates
pw = c.image_to_world(point, 0)
# needs tuning
# pw_alternative = c.R.transpose().dot(point) - c.R.transpose().dot(c.t)
pw_corners = c.image_to_world(np.array([[0, 0], [0, v], [u, 0], [u, v]]).transpose(), 0)

# Find homography
# points1 = np.zeros((4, 2), dtype=np.float32)
# points2 = np.zeros((4, 2), dtype=np.float32)
points1 = np.array([[0, 0], [0, v], [u, 0], [u, v]])
points2 = pw_corners[0:2].transpose()
# h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
h = cv2.getPerspectiveTransform(points1.astype(np.float32), points2.astype(np.float32))

workDir = 'd:\work\windpropeller'
pictureNumber = 76
# Use homography
# height, width, channels = im2.shape
orig = cv2.imread(workDir + '\snapshots\snapshot000' + str(pictureNumber) + '.jpg')
warp = cv2.warpPerspective(orig, h*100, (u, v))
cv2.imwrite(workDir + '\warped\snapshot000' + str(pictureNumber) + '.jpg', warp)


print("image to world coordinates (z = 0):" + str(pw))
plot_helper.plot_point(ax, pw, 'green')

R *= 10

plot_helper.plot_vector(ax, cameraPosition[:, 0], R[0, :], colors['green'])  # green
plot_helper.plot_vector(ax, cameraPosition[:, 0], R[1, :], colors['red'])  # red
plot_helper.plot_vector(ax, cameraPosition[:, 0], R[2, :], colors['blue'])  # violet

plot_helper.plot_plane(ax, cameraPosition, R[0], colors['green'])
plot_helper.plot_plane(ax, cameraPosition, R[1], colors['blue'])

plot_helper.plot_point(ax, [0, 0, 0], 'red')
plot_helper.plot_point(ax, cameraPosition[:, 0], 'red')

plot_helper.set_axes_equal(ax)

plt.show()
