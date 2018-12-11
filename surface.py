from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import helpers.plot_helper as plot_helper
import helpers.matrix_helper as matrix_helper
import helpers.image_helper as image_helper

import math
import numpy as np
from ODSReader import ODSReader
import helpers.camera as camera

import cv2
import logging

logging.basicConfig(level=logging.DEBUG)

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
ax = plot_helper.prepare_plot()

# pictureNumbersMin = 76
# pictureNumbersMax = 98
# n = pictureNumbersMax - pictureNumbersMin + 1
workDir = 'c:\work\windpropeller'

doc = ODSReader(u'metadata-v1_manual.ods', clonespannedcolumns=True)
table = doc.getSheet(u'1-A-PressureSide')
print(table)

img_dictionary = {}
img_params = {'x_min': 999999, 'y_min': 999999, 'x_max': -999999, 'y_max': -999999}

for i in range(1, len(table)):
    fileName = table[i][0]
    print(fileName)
    matrix_str = table[i][1]
    matrix = np.matrix(matrix_str).reshape(-1, 4)
    print(matrix)
    matrix_arr = np.asarray(matrix)
    matrix_arr = np.vstack((matrix_arr[0], matrix_arr[2], matrix_arr[1], matrix_arr[3]))
    # # matrix_arr = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]).dot(matrix_arr)
    # print("matrix_arr")
    # print(matrix_arr)
    # basis = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # matrix_arr = basis.dot(matrix_arr)

    # R = matrix_arr[0:3, 0:3]
    # T = matrix_arr[0:3, 3:4]
    # cameraPosition = -R.transpose().dot(T)

    R = matrix_arr[0:3, 0:3]
    cameraPosition = matrix_arr[0:3, 3:4]
    T = -R.dot(cameraPosition)
    plot_helper.plot_camera(ax, cameraPosition, R, 5)

    # R = R.transpose()
    # cameraPosition = T
    # T = -R.dot(cameraPosition)
    # plot_helper.plot_camera(ax, cameraPosition, R, 5)

    # cameraPosition = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]).dot(cameraPosition)
    # R = matrix_helper.getRotationMatrix(-math.pi / 2, 0, 0).dot(R)
    # T = -R.dot(cameraPosition)
    # plot_helper.plot_camera(ax, cameraPosition, R, 5)

    logging.info("cameraPosition: " + str(cameraPosition))
    logging.info("R:" + str(R))
    logging.info("T:" + str(T))

    u, v = 5456, 3632
    camera_half_angle = (38.2 / 2) * math.pi / 180
    f = u / 2 / math.tan(camera_half_angle)
    c = camera.Camera()
    c.set_K_elements(u / 2, v / 2, f=f)
    c.set_R(R)
    c.set_t(T)

    warp, img_corners = image_helper.apply_homography(c, u, v, workDir=workDir,
                                                      pictureFilename=fileName, scale=50, plane=np.array([0, 0, 1, 0]))
    left_top_y, left_top_x = img_corners[0, 0].astype(int), img_corners[1, 0].astype(int)
    img_dictionary[fileName] = {'img': warp, 'left_top_x': left_top_x, 'left_top_y': left_top_y}
    img_params['x_min'] = min(img_params['x_min'], left_top_x)
    img_params['y_min'] = min(img_params['y_min'], left_top_y)
    img_params['x_max'] = max(img_params['x_max'], left_top_x + warp.shape[0])
    img_params['y_max'] = max(img_params['y_max'], left_top_y + warp.shape[1])

background = np.zeros((img_params['x_max'] - img_params['x_min'], img_params['y_max'] - img_params['y_min'], 3),
                      np.uint8)
for key in img_dictionary:
    warp = img_dictionary[key]['img']
    left_top_x = img_dictionary[key]['left_top_x'] - img_params['x_min']
    left_top_y = img_dictionary[key]['left_top_y'] - img_params['y_min']

    alpha_s = warp[:, :, 3] / 255.0 - 0.5
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        background[left_top_x:left_top_x + warp.shape[0], left_top_y:left_top_y + warp.shape[1], c] = \
            warp[:, :, c] * alpha_s + \
            background[left_top_x:left_top_x + warp.shape[0], left_top_y:left_top_y + warp.shape[1], c] * alpha_l

cv2.imwrite(workDir + "/background.png", background)
plot_helper.set_axes_equal(ax)
plt.show()
