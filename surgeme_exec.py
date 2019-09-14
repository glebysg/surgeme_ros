import logging
import time
import os
import unittest
import numpy as np
import copy
import sys
import random
from darknet_ros_msgs.msg import BoundingBoxes
from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
import pickle as pkl
import csv
import IPython
import rospy
import cv2
import itertools
from std_msgs.msg import String
from surgeme_models import *
from helpers import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs
from yumi_homography_functions import *
from sensor_msgs.msg import CameraInfo
from scipy.spatial import distance
import math
from scipy.interpolate import interp1d
from mrcnn_msgs.msg import ObjMasks
import math
import matplotlib.pyplot as plt

class Scene():
	def __init__(self):
		self.pegs = []
		self.poles_found = []
		self.bridge = CvBridge()
		self.KI = []
		self.depth_vals = []
		self.color_frame =[]
		self.mask_frame =[]
		self.pole_flag = 0


	def get_bbox(self,data):
		# print("Entered")
		pegs = []
		count = 0
		bounds = data.bounding_boxes
		# print(data.bounding_boxes[0] )
		for box in data.bounding_boxes:
			if box.Class == "peg":
				# print("ENtered")
				pegs.append([box.xmin, box.xmax, box.ymin, box.ymax])
		self.pegs = pegs
		# poles = []

		# This is the section to detect poles, it should be configured to run once only
		# for box in data.bounding_boxes:
		# 	if box.Class == "pole":
		# 		count = count+1

		# if count == 12:
		# 	for box in data.bounding_boxes:
		# 		if box.Class == "pole":
		# 			poles.append([box.xmin, box.xmax, box.ymin, box.ymax])				
		# 	self.pole = poles
		# 	count = 0
		# 	print(len(self.pole))
		# 	print self.pole
		# count = 0
		



	def depth_cb(self,data):
	#img = data
	#rospy.loginfo(img.encoding)
		try:
			data.encoding = "mono16"
			cv_image = self.bridge.imgmsg_to_cv2(data, "mono16")
		except CvBridgeError as e:
			print(e)

		(rows,cols) = cv_image.shape
		self.depth_vals = cv_image/1000.0
		# print(depth_vals.shape)


	def camera_callback(self, data):

		# print(data.K)
		# data.encoding = "bgr8"
		self.K = np.array(list(data.K)).reshape(3,3)

	def image_callback(self, data):

		color_frame = self.bridge.imgmsg_to_cv2(data, "rgb8")
		self.color_frame = cv2.cvtColor(color_frame,cv2.COLOR_BGR2RGB)
		# cv2.imshow('image',self.color_frame)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()	

	def mask_callback(self,data):
		
		pegs = []
		for box in data.bbs:
			pegs.append([box.data[0], box.data[2], box.data[1], box.data[3]])
		self.pegs = pegs
		# mask_img = data.masks
		# print(mask_img.encoding)

		if len(data.masks[0].data)>0:
			mask_frame = self.bridge.imgmsg_to_cv2(data.masks[0],'passthrough')
			self.mask_frame = cv2.cvtColor(mask_frame,cv2.COLOR_BGR2RGB)
			# print self.mask_frame.shape
		# cv2.imshow("MASK",self.mask_frame)
		# cv2.waitKey(0)

	def pole_cb(self,data):
		poles = []
		count = 0
		for pole in data.bounding_boxes:

			if pole.Class == "pole"+str(count):

				poles.append([(pole.xmin+pole.xmax)/2,(pole.ymin+pole.ymax)/2])
				count = count+1
		count = 0
		self.poles_found = np.array(poles)
		# print self.poles_found
		self.pole_flag = 1





########################### ENd of all callback functions #########################



	############### Stuff to display the depth image#########################

	# #cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
 #  # Normalize the depth image to fall between 0 (black) and 1 (white)
 #  # http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
 #  	#cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
	# cv_image =cv2.convertScaleAbs(cv_image, alpha=0.05)
	# cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
	# cv_image_norm = cv_image
	# #print(cv_image_norm)
	# if cols > 60 and rows > 60 :
	# 	cv2.circle(cv_image, (50,50), 10, 255)

	# 	cv2.imshow("Image window", cv_image_norm)
	# 	cv2.waitKey(3)

	def grasp_points(self,ROI,bbox, offset=1):
		gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)
		# dst = cv2.cornerHarris(gray,5,9,0.19)
		# dst = cv2.dilate(dst,None)
		# print dst.shape
		h,w = gray.shape
		h = h/2
		w = w/2

		cor = cv2.goodFeaturesToTrack(gray,3,0.005,50)
		print(cor)
		cor = np.int0(cor)
		corners = []
		# print(cor)
        
	

		for i in cor:
			x,y = i.ravel()
			corners.append([x,y])
			cv2.circle(ROI,(x,y),3,255,-1)



		corners = np.array(corners)
		pole = get_center_triangle(corners)
		
		# print ("Triangular Corners:",corners)
		#print ("Pole :",pole)
		# ###################### Finding Grasp Points
		gpoints = []
		corner_pairs = np.array([[[0,0],[0,0]]])
		# Get the actual grasping points, inside the object of interest
		for point1, point2 in list(itertools.combinations(corners, 2)):
			
			center = np.array([abs(point1[0]+point2[0])/2, abs(point1[1]+point2[1])/2])
			corner_pairs = np.concatenate((corner_pairs,np.array([[point1,point2]])),axis = 0)
			gpoint_plus = center + normal_to_line(point1,point2)*offset
			gpoint_minus = center - normal_to_line(point1,point2)*offset
			if np.linalg.norm(pole-abs(gpoint_plus))<np.linalg.norm(pole-abs(gpoint_minus)):
				gpoints.append(gpoint_plus.astype(int))

			else:
				gpoints.append(gpoint_minus.astype(int))
		gpoints = np.array(gpoints)
		print("Grasp Points",gpoints)

		for coords in gpoints:
			cv2.circle(ROI,(coords[0],coords[1]),3,(0,255,0),-1)
		cv2.imshow("Corners",ROI)
		cv2.waitKey(0)
		grasp_point,grasp_index,corner_point,corner_index = min_dist_robot2points([0,w],gpoints,corners)#h,0 for left hand or h and w*2 for right hand
		# print grasp_point
		grip_corners = corner_pairs[grasp_index+1]
		angle = self.grip_angle(grip_corners)
		grasp_point[0] = grasp_point[0]+bbox[2]
		grasp_point[1] = grasp_point[1]+bbox[0]
		corner_point[0] = corner_point[0]+bbox[2]
		corner_point[1] = corner_point[1]+bbox[0]
		
		return np.array(grasp_point),angle,np.array(corner_point)

	def grip_angle(self,grip_corners):
		print ("Grip Corners:",grip_corners)
		corner1 = grip_corners[0]
		corner2 = grip_corners[1]
		dY = (corner1[1]-corner2[1])
		dX = (corner1[0]-corner2[0])
		print("Dx, Dy", dX, dY)
		angle = math.atan(dY/dX)
		angle = math.degrees(angle)
		
		if dX<0 and dY<0 :
			angle = angle
		elif dX<0 or dY<0:
			angle = angle+180 
	

		print ("Orientation Angle:",(angle))
		return angle


	def drop_pose(self,ROI,limb,bbox,pole_pos):

		arm = opposite_limb if limb == limb else limb
		gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)

		h,w = gray.shape
		
		

		cor = cv2.goodFeaturesToTrack(gray,5,0.005,35)
		cor = np.int0(cor)
		corners = []
		for i in cor:
			x,y = i.ravel()
			corners.append([x,y])
			# cv2.circle(ROI,(x,y),3,255,-1)

		corners = np.array(corners)
		#corner_point is the opposite corner point in pixels
		corner_point,corner_index = min_dist_robot2drop_pt([0,w/2],corners)#h,0 for left hand or h and w*2 for right hand

		cv2.circle(ROI,(corner_point[0],corner_point[1]),3,(0,255,0),-1)
		rob_pose = execution.get_curr_pose(opposite_limb)
		# print rob_pose
		cpoint = [0,0]
		cpoint[0] = corner_point[0]+bbox[2]
		cpoint[1] = corner_point[1]+bbox[0]
		cpoint = np.array(cpoint)		
		# Caluclate depth based on pixels that are mask segmented 

		segment_pts = np.argwhere(gray>250)
		segment_pts = segment_pts+np.array([bbox[2],bbox[0]])
		counter = 0
		sum_vals = 0
		for y,x in segment_pts:
			if self.depth_vals[x,y]>0.0:
				sum_vals = sum_vals+self.depth_vals[x,y]
				counter = counter+1

				
	
		z_coord = sum_vals/counter
		# print z_coord
		#corner_robot_pose is the opposite corner in robot space
		corner_robot_pose = cam2robot(cpoint[0], cpoint[1], z_coord, scene.K,opposite_limb)
		# dX = (corner_robot_pose[0]+rob_pose.translation[0])/2
		center_coord_x = (rob_pose.translation[0])+(corner_robot_pose[0]-rob_pose.translation[0])/2
		center_coord_y = (rob_pose.translation[1])+(corner_robot_pose[1]-rob_pose.translation[1])/2
		# dY = (corner_robot_pose[1]+rob_pose.translation[1])/2
		print("gripper pose",rob_pose.translation[0],rob_pose.translation[1])
		print("corner pose",corner_robot_pose[0],corner_robot_pose[1])
		# pole_pose=[0.315,-0.075]
		pole_pose = [pole_pos[0],pole_pos[1]]
		movetodelta=[pole_pose[0]-center_coord_x,pole_pose[1]-center_coord_y]
		print ("it will move to delta",movetodelta)
		cv2.imshow("Drop_Corners",ROI)
		cv2.waitKey(0)

		return movetodelta

	def pole_positions_rob(self,pole_poses):

		left_ids = [0,2,3,6,7,10]
		poles_left = []
		right_ids = [1,4,5,8,9,11]
		poles_right = []
		poles_xyz=[]

		for x,y in pole_poses:
			cv2.circle(self.color_frame,(x,y),3,(0,255,0),-1)
			poles_xyz.append([x,y,self.depth_vals[x,y]])
		poles_xyz = np.array(poles_xyz)
		print poles_xyz
		for i in left_ids:
			poles_left.append(poles_xyz[i])
		poles_left = np.array(poles_left)
		poles_left_arm = cam2robot_array(poles_left,self.K,limb)

		for i in right_ids:
			poles_right.append(poles_xyz[i])
		poles_right = np.array(poles_right)
		poles_right_arm = cam2robot_array(poles_right,self.K,opposite_limb)

		cv2.imshow("Poles",self.color_frame)
		cv2.waitKey(0)
		return poles_left_arm,poles_right_arm


	def subscribe(self):
		rospy.init_node('BoundingBoxes', anonymous=True)
		# rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.get_bbox) #This is tthe subscriber for the darknet bounding boxes. SInce we use mask we dont need this
		rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb)
		rospy.Subscriber("/camera/color/camera_info",CameraInfo, self.camera_callback)
		rospy.Subscriber("/camera/color/image_raw",Image, self.image_callback)
		rospy.Subscriber("/masks_t",ObjMasks,self.mask_callback)
		rospy.Subscriber("/darknet_ros/tracked_bbs",BoundingBoxes,self.pole_cb)

##########################################################################################################################

def get_max_depth(depth_mat):

######### Get the max depth index value within a given ROI ############
	max_depth = np.amin(depth_mat)
	if max_depth < 0.3:
		print("Changing Depth")
		max_depth = 0.33
	return max_depth

def get_min_depth(depth_mat):

######### Get the max depth index value within a given ROI ############
	min_depth = np.amax(depth_mat)
	if min_depth > 0.36:
		print("Changing Depth")
		min_depth = 0.35
	return min_depth

def get_avg_depth(depth_mat):

	# pts = np.argwhere((depth_mat<0.300) & (depth_mat>0.0))
	# # pts = np.argwhere(depth_mat[pts])
	# sum_val = 0
	# for x,y in pts:
	# 	# print depth_mat[x,y]
	# 	sum_val = sum_val+depth_mat[x,y]
	# avg_val = sum_val/len(pts)
	# return avg_val
	print depth_mat



def min_dist_robot2points(rob,pts,cpts ):

	dist = []
	for i in pts:
		dist.append(np.linalg.norm(i - rob))
	min_pt = np.argmin(dist)

	distc = []
	for i in cpts:
		distc.append(np.linalg.norm(i - rob))
	min_ptc = np.argmin(distc)
	# print min_pt
	return pts[min_pt],min_pt,cpts[min_ptc],min_ptc

def min_dist_robot2drop_pt(rob,cpts ):

	distc = []
	for i in cpts:
		distc.append(np.linalg.norm(i - rob))
	min_ptc = np.argmin(distc)
	# print min_pt
	return cpts[min_ptc],min_ptc



if __name__ == '__main__':

	# Start Robot Environment
	# Start surgemes class and open grippers and take robot to neutral positions
	execution = Surgemes()
	time.sleep(1)
	limb = input('Ente the limb : ')
	opposite_limb = 'right' if limb == 'left' else 'left'

	rob_pose = execution.get_curr_pose(limb)
	execution.left_open()
	execution.right_open()
	time.sleep(2)
	execution.ret_to_neutral_angles(limb)
	execution.ret_to_neutral_angles(opposite_limb)
	time.sleep(5)



	ROI_offset = 5 #ROI bounding box offsets
	# Start the Scene calss to obtain pole positions and bounding boxes etc. 
	scene = Scene()
	scene.subscribe()
	# # rospy.spin()
	time.sleep(2)
	while scene.pole_flag == 0:
		a = 1
	print(scene.poles_found)
	left_poles,right_poles = scene.pole_positions_rob(scene.poles_found)


	# # exit()rostop
	while len(scene.pegs) == 0:
		a = 1
	first_peg = np.array(scene.pegs) #xmin,xmax,ymin,ymax
	first_peg = first_peg.reshape(4)
	first_peg[0] = first_peg[0]-ROI_offset
	first_peg[2] = first_peg[2]-ROI_offset
	first_peg[1] = first_peg[1]+ROI_offset
	first_peg[3] = first_peg[3]+ROI_offset
	print first_peg
	# ROI = scene.color_frame[first_peg[2]-10:first_peg[3]+10,first_peg[0]-10:first_peg[1]+10,:]
	while len(scene.mask_frame) == 0:
		a = 1	
	# print scene.mask_frame

	# print(first_peg[0][0])
	ROI = scene.mask_frame[first_peg[0]:first_peg[1],first_peg[2]:first_peg[3],:] #xmin,xmax,ymin,ymax
	depth_ROI = scene.depth_vals[first_peg[0]-10:first_peg[1]+10,first_peg[2]-10:first_peg[3]+10]
	# depth_ROI = scene.depth_vals[first_peg[3]-10:first_peg[2]+10,first_peg[1]-10:first_peg[0]+10]
	z_coord = get_max_depth(depth_ROI)

	bbox_corners = create_bbox_rob_points(first_peg,z_coord)

	yumi_poses = cam2robot_array(bbox_corners,scene.K,limb)

	
	grasp,g_angle,corner = scene.grasp_points(ROI,first_peg)
	print grasp
	execution.joint_orient(limb,g_angle)
	x = grasp[0]+first_peg[2]
	y = grasp[1]+first_peg[0]
	align = np.array([x,y])
	cx = corner[0]+first_peg[2]
	cy = corner[1]+first_peg[0]
	calign = np.array([cx,cy])

	yumi_pose = cam2robot(calign[0], calign[1], z_coord , scene.K,limb)
	yumi_pose[2] = 0.07
	execution.surgeme1(1,yumi_pose,limb)
	time.sleep(5)
	print("Finished execution")

	z_coord = get_min_depth(depth_ROI)
	print ("Z Coord for grasping",z_coord)
	yumi_pose = cam2robot(align[0], align[1], z_coord, scene.K,limb)
	yumi_pose[2] = 0.0173 #Constant depth  
	print(yumi_pose)
	# a = input("Do a depth Check ")
	execution.surgeme2(1,yumi_pose,limb)
	
	
	print("Finished Grasping")

	execution.surgeme3(1,limb)
	time.sleep(5)
	print("FInished Lift")


	execution.ret_to_neutral_angles(limb)
	time.sleep(5)
	
	execution.surgeme4(limb)
	print ("Finished Go TO Transfer ")

	execution.surgeme5(limb)
	print("Finsihed Transfer ")

	execution.ret_to_neutral_angles(limb)
	execution.ret_to_neutral_angles(opposite_limb)
	time.sleep(10)
	while len(scene.pegs) == 0:
		a = 1
	first_peg = []
	first_peg = np.array(scene.pegs) #xmin,xmax,ymin,ymax
	first_peg = first_peg.reshape(4)
	first_peg[0] = first_peg[0]-ROI_offset
	first_peg[2] = first_peg[2]-ROI_offset
	first_peg[1] = first_peg[1]+ROI_offset
	first_peg[3] = first_peg[3]+ROI_offset
	print first_peg
	# ROI = scene.color_frame[first_peg[2]-10:first_peg[3]+10,first_peg[0]-10:first_peg[1]+10,:]
	while len(scene.mask_frame) == 0:
		a = 1	
	# print scene.mask_frame

	# print(first_peg[0][0])
	ROI = scene.mask_frame[first_peg[0]:first_peg[1],first_peg[2]:first_peg[3],:] #xmin,xmax,ymin,ymax
	# choose which pole coord to send it to 
	drop_pt = scene.drop_pose(ROI,limb,first_peg,right_poles[0])
	execution.surgeme6(drop_pt,limb)
	print ("Finished execution ")
	execution.surgeme7(limb)
	print("Finished align and Drop") 
	execution.ret_to_neutral_angles(opposite_limb)

	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		exit(0)
	rospy.spin()


