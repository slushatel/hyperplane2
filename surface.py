from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ODSReader import ODSReader

doc = ODSReader(u'metadata-v1_manual.ods', clonespannedcolumns=True)
table = doc.getSheet(u'1-A-PressureSide')
print(table)
for i in range(1, len(table)):
    print(table[i][0])
    matrix_str = table[i][1]
    matrix = np.matrix(matrix_str).reshape(-1, 4)
    print(matrix)
    # print(matrix.)

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
z = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
