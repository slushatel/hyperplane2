from math import sqrt

import cv_samples.camera as camera
import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cv2


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


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


def plot_plane(ax, point, normal, color):
    x = np.linspace(point[0] - 20, point[0] + 20, 10)
    y = np.linspace(point[1] - 20, point[1] + 20, 10)

    X, Y = np.meshgrid(x, y)
    Z = - (normal[0] * (X - point[0]) + normal[1] * (Y - point[1])) / normal[2] + point[2]
    ax.plot_surface(X, Y, Z, alpha=0.2, color=color)


# print(np.dot(rotation_matrix(axis, theta), v))
######################################

c = camera.Camera()
c.set_K_elements(0, 0)
R = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]])

# R = np.dot(rotation_matrix(axis, theta), R)

for i in range(0, 3):
    len1 = sqrt(pow(R[0, i], 2) + pow(R[1, i], 2) + pow(R[2, i], 2))
    print("The length of all rotation vectors: " + str(len1))

# c.set_R(R)
c.set_R_euler_angles([math.pi / 8, 0, math.pi / 4])
R = c.R
cameraPosition = np.array([[0], [0], [50]])
translationMatrix = -R.dot(cameraPosition)
c.set_t(translationMatrix)

print("camera center: " + str(-R.transpose().dot(translationMatrix)))

point = c.world_to_image(np.array([[0., 0., 0.]]).T)
print("point's image coordinates: " + str(point))

# camera point from opencv
imagePoints = []
imagePoints = cv2.projectPoints(np.array([[0., 0., 0.]]), R, translationMatrix, c.K, np.array([]))
print("point's image coordinates (using projectPoints): " + str(imagePoints))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

#  world coordinates
pw = c.image_to_world(point, 1)
print("image to world coordinates (z = 0):" + str(pw))
ax.scatter(pw[0], pw[1], pw[2], color='green')

# c.plot_world_points(point, 'ro')


x, y, z = [], [], []

# for i in range(1, len(table)):
x.append(0)
y.append(0)
z.append(0)

x.append(cameraPosition[0, 0])
y.append(cameraPosition[1, 0])
z.append(cameraPosition[2, 0])

R *= 10

# ax.quiver(cameraPosition[0, 0], cameraPosition[1, 0], cameraPosition[2, 0],
#           R[0, 0], R[1, 0], R[2, 0], color=(0.5, 1, 0))  # green
# ax.quiver(cameraPosition[0, 0], cameraPosition[1, 0], cameraPosition[2, 0],
#           R[0, 1], R[1, 1], R[2, 1], color=(1, 0, 0.5))  # red
# ax.quiver(cameraPosition[0, 0], cameraPosition[1, 0], cameraPosition[2, 0],
#           R[0, 2], R[1, 2], R[2, 2], color=(0.5, 0, 1))  # violet

ax.quiver(cameraPosition[0, 0], cameraPosition[1, 0], cameraPosition[2, 0],
          R[0, 0], R[0, 1], R[0, 2], color=(0.5, 1, 0))  # green
ax.quiver(cameraPosition[0, 0], cameraPosition[1, 0], cameraPosition[2, 0],
          R[1, 0], R[1, 1], R[1, 2], color=(1, 0, 0.5))  # red
ax.quiver(cameraPosition[0, 0], cameraPosition[1, 0], cameraPosition[2, 0],
          R[2, 0], R[2, 1], R[2, 2], color=(0.5, 0, 1))  # violet

plot_plane(ax, cameraPosition, R[0], colors['green'])
plot_plane(ax, cameraPosition, R[1], colors['blue'])

# ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[0, 0], matrix[0, 1], matrix[0, 2])
# ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[1, 0], matrix[1, 1], matrix[1, 2])
# ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[2, 0], matrix[2, 1], matrix[2, 2])
#
# ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[0, 0], matrix[1, 0], matrix[2, 0], color=(0.5, 1, 0)) #green
# ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[0, 1], matrix[1, 1], matrix[2, 1], color=(1, 0, 0.5)) #red
# ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[0, 2], matrix[1, 2], matrix[2, 2], color=(0.5, 0, 1)) #violet

# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
# z = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

set_axes_equal(ax)

plt.show()
