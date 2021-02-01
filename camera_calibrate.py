# this is to calibrate the camera to obtain
# the camera matrix and distortion coeff.
# and then save to 'cali_points.json'

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

nx = 9
ny = 6
cal_dir = 'camera_cal/'     # directory of pictures for calibration
json_path = 'camera_cali_result.json'

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
img = None

# loop all calibration pictures from cal_dir
for fname in os.listdir(cal_dir):
    fname = cal_dir + fname
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)

# write camera matrix and dist coeff to json file
j = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
}

with open(json_path, "w") as write_file:
    json.dump(j, write_file, indent = 2)