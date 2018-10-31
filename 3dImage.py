import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
import numpy as np
from PIL import Image

# im = Image.open('c:\work\windpropeller\snapshots\snapshot00076.jpg')
# im.save('c:\work\windpropeller\snapshots\snapshot00076.png')

fn = get_sample_data('c:\work\windpropeller\snapshots\snapshot00076.png', asfileobj=False)
img = read_png(fn)

x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]

ax = plt.gca(projection='3d')
ax.plot_surface(x, y, 0*x + 0*y, rstride=20, cstride=20,
                facecolors=img)
plt.show()
