import cv2
import numpy as np

yaw = 0
pitch = 0
roll = 0
w, h = 600, 300
A = np.matrix([
    [1, 0, -w / 2],
    [0, 1, -h / 2],
    [0, 0, 0],
    [0, 0, 1]])

R_y = np.matrix([
    [1, 0, 0, 0],
    [0, np.cos(yaw), -np.sin(yaw), 0],
    [0, np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 0, 1]])

R_p = np.matrix([
    [np.cos(pitch), 0, -np.sin(pitch), 0],
    [0, 1, 0, 0],
    [np.sin(pitch), 0, np.cos(pitch), 0],
    [0, 0, 0, 1]])
R_r = np.matrix([
    [np.cos(roll), -np.sin(roll), 0, 0],
    [np.sin(roll), np.cos(roll), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])

R = R_y * R_p * R_r

T = np.matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, -1.50],
    [0, 0, 0, 1]])
f = w / (2 * np.tan(90 * np.pi / 360))
K = np.matrix([
    [f, 0, w / 2, 0],
    [0, f, h / 2, 0],
    [0, 0, 1, 0], ])

H = K * (T * (R * A))

orig = cv2.imread('XXX.png')
warp = cv2.warpPerspective(orig, H, (600, 300))
