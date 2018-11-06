from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ODSReader import ODSReader


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


doc = ODSReader(u'metadata-v1_manual.ods', clonespannedcolumns=True)
table = doc.getSheet(u'1-A-PressureSide')
print(table)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

x, y, z = [], [], []

# for i in range(1, len(table)):
for i in range(1, 5):
    print(table[i][0])
    matrix_str = table[i][1]
    matrix = np.matrix(matrix_str).reshape(-1, 4)
    print(matrix)
    x.append(matrix[0, 3])
    y.append(matrix[1, 3])
    z.append(matrix[2, 3])

    ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[0, 0], matrix[0, 1], matrix[0, 2])
    ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[1, 0], matrix[1, 1], matrix[1, 2])
    ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[2, 0], matrix[2, 1], matrix[2, 2])

    ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[0, 0], matrix[1, 0], matrix[2, 0], color=(0.5, 1, 0)) #green
    ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[0, 1], matrix[1, 1], matrix[2, 1], color=(1, 0, 0.5)) #red
    ax.quiver(matrix[0, 3], matrix[1, 3], matrix[2, 3], matrix[0, 2], matrix[1, 2], matrix[2, 2], color=(0.5, 0, 1)) #violet

# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
# z = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

set_axes_equal(ax)

plt.show()

'''
def get_test_data(delta=0.05):

    from matplotlib.mlab import  bivariate_normal
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)

    Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = Z2 - Z1

    X = X * 10
    Y = Y * 10
    Z = Z * 500
    return X, Y, Z

'''

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = axes3d.get_test_data(0.05)
# ax.plot_wireframe(x, y, z, rstride=2, cstride=2)
# plt.show()
