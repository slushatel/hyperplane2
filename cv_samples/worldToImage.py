import helpers.camera as camera
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import helpers.plot_helper as plot_helper
import helpers.image_helper as image_helper

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
ax = plot_helper.prepare_plot()

u, v = 5456, 3632
camera_half_angle = math.pi * 38.2 / 180
f = u / 2 / math.tan(camera_half_angle)
c = camera.Camera()
c.set_K_elements(u / 2, v / 2, f=f)

cameraPosition = np.array([[-2.17281528],
                           [-0.43091152],
                           [9.74970592]])
R = np.array([[0.9973, -0.0729, 0.0012],
              [-0.0547, -0.7587, -0.6491],
              [0.0482, 0.6473, -0.7607]])
translationMatrix = -R.dot(cameraPosition)
c.set_R(R)
c.set_t(translationMatrix)
# cameraPosition = -R.transpose().dot(translationMatrix)
print("camera center: " + str(cameraPosition))

# image_helper.apply_homography(c, u, v, workDir='c:\work\windpropeller', pictureFilename='snapshot00076.jpg', scale=10,
#                               z=0)

image_helper.apply_homography(c, u, v, workDir='c:\work\windpropeller', pictureFilename='snapshot00076.jpg', scale=10,
                              plane=np.array([0,-1,1,2]), ax = ax)
# image_helper.apply_homography(c, u, v, workDir='c:\work\windpropeller', pictureFilename='snapshot00076.jpg', scale=10,
#                               plane=np.array([1, 1, 1, -3]), ax=ax)
plot_helper.plot_camera(ax, cameraPosition[:, 0], R, 5)

plot_helper.plot_point(ax, [0, 0, 0], 'red')
plot_helper.set_axes_equal(ax)
plt.show()
