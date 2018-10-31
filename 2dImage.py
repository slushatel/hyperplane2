import cv2
import numpy as np
# import matplotlib

# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import logging
# from PyQt5 import QtWidgets
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

logging.basicConfig(level=logging.DEBUG)

workDir = 'c:\work\windpropeller'

pictures = []
pictureNumbersMin = 76
pictureNumbersMax = 78
i = 0
n = pictureNumbersMax - pictureNumbersMin + 1
background = np.zeros((4000*n, 6000, 4), np.uint8)
for i in range(0, n):
    pictureNumber = pictureNumbersMax - i
    img = cv2.imread(workDir + '\snapshots\snapshot000' + str(pictureNumber) + '.jpg', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    plt.subplot2grid((n, 2), (i, 0))
    plt.title('Input ' + str(pictureNumber))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # i = i + 1
    pictures.append(img)

logging.info('files loaded')

for idx, img in enumerate(pictures):
    pts1 = np.float32([[0, 0], [6000, 0], [0, 6000], [6000, 6000]])
    pts2 = np.float32([[0, 0], [6000, 500], [0, 5000], [6000, 6000]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (6000, 4000))
    alpha_s = dst[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        background[4000 * idx:4000 * (idx + 1), 0:6000, c] = dst[:, :, c] * alpha_s + background[4000 * idx:4000 * (idx + 1), 0:6000, c] * alpha_l

logging.info('background created')

cv2.imwrite(workDir + "/background.png", background)
logging.info('background saved')

plt.subplot2grid((n, 2), (0, 1), rowspan=n)
plt.title('Output')
plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))

# plt.show()
cv2.namedWindow("win0", cv2.WINDOW_GUI_NORMAL)
cv2.imshow("win0", background)
cv2.waitKey(5000)
