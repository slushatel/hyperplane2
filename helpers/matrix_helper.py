import numpy as np
import math


def getRotationMatrix(angle_x, angle_y, angle_z):
    Rx = np.array([[1, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x)], [0, math.sin(angle_x), math.cos(angle_x)]])
    Ry = np.array([[math.cos(angle_y), 0, -math.sin(angle_y)], [0, 1, 0], [math.sin(angle_y), 0, math.cos(angle_y)]])
    Rz = np.array([[math.cos(angle_z), -math.sin(angle_z), 0], [math.sin(angle_z), math.cos(angle_z), 0], [0, 0, 1]])
    return np.dot(Rx, np.dot(Ry, Rz))
