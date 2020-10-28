## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
from os.path import join
from glob import glob
import pickle
from pprint import pprint as pp
from yumi_helpers import get_3dpt_depth, rigid_transform_3D
from math import sqrt
from yumi_homography import affine_matrix_from_points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Start streaming
pipeline = None
count = 0
img_path ='./data/'
homography_path = "homography.txt"
# Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
# pipe_profile = pipeline.start(config)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#########################
#     CALIBRATION       #
#########################
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
x_grid = 9
y_grid = 6
images = glob(join(img_path,'*.jpg'))
depth = glob(join(img_path,'*.npy'))
images = filter(lambda f: 'img' in f, images)
depth = filter(lambda f: 'depth' in f, depth)
images.sort()
depth.sort()
pixels = []
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d image points.
# get intrinsic color camera params
# frames = pipeline.wait_for_frames()
# color_frame = frames.get_color_frame()
# color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
color_intrin = np.load(join(img_path, 'intrinsics.npy'))
selected_coords = [[[0,0], [2,2], [1,5], [5,1], [3,4], [3,8]],
                   [[0,4], [2,1], [1,7], [4,3], [5,6], [3,7]]]

# pipeline.stop()

for iname, dname, selected_coords in zip(images, depth, selected_coords):
    # Arrays to store object points and image points from all the images.
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # We start with the bottom floor
    objp = np.zeros((y_grid,x_grid,3), np.float32)
    for i in range(y_grid):
        for j in range(x_grid):
            objp[i,j,1] = i*2.45
            objp[i,j,0] = j*2.45
    # Divide all the points by 100 so they are in meters
    objp = objp/100.0
    # add the height to the second image
    if "_img1" in iname:
        objp[:,:,2] = np.ones((y_grid,x_grid))*-0.05562
    # Flatten the array
    objp = np.array(objp, dtype=np.float64).reshape(-1,1,3).tolist()
    # get the color image
    img = cv2.imread(iname)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # get the depth image
    depth = np.load(dname)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (x_grid,y_grid),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        pixel_matrix = np.array(corners2).reshape((y_grid,x_grid,1,2))
        world_p = np.array(objp)
        for coord in selected_coords:
            # get the depth point
            world_p.shape
            world = world_p[coord[0]*x_grid+coord[1]][0]
            col, row = pixel_matrix[coord[0],coord[1]][0]
            col = int(col)
            row = int(row)
            z = depth[row,col]
            # skip the point if the depth makes no sense
            if z > 1:
                continue
            # get the pointcloud point
            pointcloud = get_3dpt_depth([col,row],z,color_intrin)
            # append the pointcloud to the image array
            imgpoints.append(pointcloud)
            # append the world point to the obj array
            objpoints.append(world)
        img = cv2.drawChessboardCorners(img, (x_grid,y_grid), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)
# test camera calibration
# create the rigid transform
imgpoints = np.array(imgpoints,dtype='float32').reshape(-1,3)
objpoints= np.array(objpoints,dtype='float32').reshape(-1,3)
# print(imgpoints.shape)
# print(objpoints.shape)

# for img, world in zip(imgpoints, objpoints):
    # print("img", img)
    # print("world", world)
    # print("/////////////////////")
print(imgpoints.T)
print(objpoints.T)
H = affine_matrix_from_points(imgpoints.T, objpoints.T, scale=True)
# print(H)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(objpoints[:,0], objpoints[:,1], objpoints[:,2], marker='^')
ax.scatter(imgpoints[:,0], imgpoints[:,1], imgpoints[:,2], marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


# for source, dest in zip(imgpoints,objpoints):
    # print("image :", source)
    # print("object:", dest)
    # print("//////////////")
# Save homography
np.savetxt(join(img_path,homography_path),H)
print(imgpoints.shape)
print(objpoints.shape)
error = []
estimated_pts = []
for source, dest in zip(imgpoints,objpoints):
    estimated_dest = np.dot(H,np.concatenate((source,[1])))
    estimated_pts.append(estimated_dest)
    # print(estimated_dest, source)
    error.append(np.linalg.norm(dest-estimated_dest[:3]))
estimated_pts = np.array(estimated_pts)
print("AVERAGE EUCLIDEAN ERROR IN METERS:", np.mean(error))
ax.scatter(estimated_pts[:,0], estimated_pts[:,1], estimated_pts[:,2], marker='*')
plt.show()
