from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import helpers.plot_helper as plot_helper

import numpy as np
import pandas as pd
from ODSReader import ODSReader

import cv2

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
ax = plot_helper.prepare_plot()
ax.invert_yaxis()

doc = ODSReader(u'metadata-v1_manual.ods', clonespannedcolumns=True)
table = doc.getSheet(u'1-A-PressureSide')
print(table)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')

x, y, z = [], [], []

for i in range(1, len(table)):
    # for i in range(1, 6):
    print(table[i][0])
    matrix_str = table[i][1]
    matrix = np.matrix(matrix_str).reshape(-1, 4)
    print(matrix)
    matrix_arr = np.asarray(matrix)
    R = matrix_arr[0:3, 0:3]
    # R = R.dot(np.diag([1, 1, -1]))
    T = matrix_arr[0:3, 3:4]
    cameraPosition = -R.transpose().dot(T)

    # cameraPosition = np.diag([1, 1, -1]).dot(cameraPosition)
    # T = -R.dot(cameraPosition)

    # camera axises
    mult = 5
    plot_helper.plot_vector(ax, cameraPosition, R[0] * mult, colors['green'])  # green
    plot_helper.plot_vector(ax, cameraPosition, R[1] * mult, colors['red'])  # red
    plot_helper.plot_vector(ax, cameraPosition, R[2] * mult, colors['blue'])  # blue

plot_helper.set_axes_equal(ax)

plt.show()

# '''
# def get_test_data(delta=0.05):
#
#     from matplotlib.mlab import  bivariate_normal
#     x = y = np.arange(-3.0, 3.0, delta)
#     X, Y = np.meshgrid(x, y)
#
#     Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
#     Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
#     Z = Z2 - Z1
#
#     X = X * 10
#     Y = Y * 10
#     Z = Z * 500
#     return X, Y, Z
#
# '''

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = axes3d.get_test_data(0.05)
# ax.plot_wireframe(x, y, z, rstride=2, cstride=2)
# plt.show()
