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


class Scene():
	def __init__(self):
		self.pegs = []
		self.bridge = CvBridge()
		self.KI = []
		self.depth_vals = []
		self.color_frame =[]


	def get_bbox(self,data):
		# print("Entered")
		pegs = []
		for box in data.bounding_boxes:
			pegs.append([box.xmin, box.xmax, box.ymin, box.ymax])
		self.pegs = pegs


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

	def grasp_points(self,ROI, offset=4):
		gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)
		# dst = cv2.cornerHarris(gray,5,9,0.19)
		# dst = cv2.dilate(dst,None)
		# print dst.shape
		h,w = gray.shape
		h = h/2
		w = w/2
		

		cor = cv2.goodFeaturesToTrack(gray,4,0.01,5)
		cor = np.int0(cor)
		corners = []
		# print(cor)
        
		####################### for obtaining corners and drawing cornr circles 
		for i in cor:
			# print ("Corners",i)
			x,y = i.ravel()
			dist = np.linalg.norm(np.array([x,y])-np.array([h,w]))
			if dist>15:
				corners.append([x,y])
				cv2.circle(ROI,(x,y),3,255,-1)
			else:
				pole = np.array([x,y])
		# print np.array(corners)

		corners = np.array(corners)
		
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
		grasp_point,grasp_index = min_dist_robot2points([0,w],gpoints)#h,0 for left hand or h and w*2 for right hand
		# print grasp_point
		grip_corners = corner_pairs[grasp_index+1]
		angle = self.grip_angle(grip_corners)
		
		return np.array(grasp_point),angle

	def grip_angle(self,grip_corners):
		print ("Grip Corners:",grip_corners)
		corner1 = grip_corners[0]
		corner2 = grip_corners[1]
		dY = (corner1[1]-corner2[1])
		dX = (corner1[0]-corner2[0])
		angle = math.atan2(dY,dX)
		angle = math.degrees(angle)
		
		if dX>0 and dY<0:
			angle = angle + 180 
	

		print ("Orientation Angle:",(angle))
		return angle


		


	def subscribe(self):
		rospy.init_node('BoundingBoxes', anonymous=True)
		rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.get_bbox)
		rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb)
		rospy.Subscriber("/camera/color/camera_info",CameraInfo, self.camera_callback)
		rospy.Subscriber("/camera/color/image_raw",Image, self.image_callback)

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


def min_dist_robot2points(rob,pts):

	dist = []
	for i in pts:
		dist.append(np.linalg.norm(i - rob))
	min_pt = np.argmin(dist)
	# print min_pt
	return pts[min_pt],min_pt






if __name__ == '__main__':
	# t = S3()
	# t.left_open()
	# t.right_open()

	scene = Scene()
	scene.subscribe()
	time.sleep(2)
	first_peg = scene.pegs[0] #xmin,xmax,ymin,ymax
	print first_peg
	ROI = scene.color_frame[first_peg[2]-10:first_peg[3]+10,first_peg[0]-10:first_peg[1]+10,:]
	depth_ROI = scene.depth_vals[first_peg[2]-10:first_peg[3]+10,first_peg[0]-10:first_peg[1]+10]
	
	z_coord = get_max_depth(depth_ROI)

	bbox_corners = create_bbox_rob_points(first_peg,z_coord)

	yumi_poses = cam2robot_array(bbox_corners,scene.K,'left')

	# print (yumi_poses)


	
	# ###################################### Approach 
	
		
	
	# time.sleep(2)


	approach = S1()
	time.sleep(1)
	rob_pose = approach.get_curr_pose('left')
	# print rob_pose
	approach_pt,approach_index = min_dist_robot2points(rob_pose.translation,yumi_poses)
	print approach_pt
	approach.left_open()
	time.sleep(2)
	approach.ret_to_neutral('left')
	approach.ret_to_neutral('right')
	time.sleep(5)
	# print "Arms at neutral"
	# # Wait until a peg is found
	grasp,g_angle = scene.grasp_points(ROI)
	print grasp
	approach.joint_orient('left',g_angle)
	x = grasp[0]+first_peg[0]-10
	y = grasp[1]+first_peg[2]-10
	align = np.array([x,y])
	# # Get the camera coords of the object of interest
	# cam_points = [first_peg[0], first_peg[2], 0.3338,1]
	# print (z_coord)
	# yumi_pose = cam2robot(align[0], align[1], approach_pt[2], scene.K,'left')

	approach.surgeme1(1,approach_pt,'left')
	time.sleep(5)
	print("Finished Approach")
	# print ROI.shape
	############################### Align and Grasp
	 
	# grasp,g_angle = scene.grasp_points(ROI)
	# print g_angle
	# approach.joint_orient('left',g_angle)

	
	
	# for coords in align:
	# 	cv2.circle(scene.color_frame,(coords[0],coords[1]),3,(0,255,0),-1)

	print("Grasp Points : ",align)
		# approach.left_open()
	# time.sleep(2)

	# # Get the camera coords of the object of interest
	# #cam_points = [first_peg[0], first_peg[2], 0.3338,1]
	z_coord = get_min_depth(depth_ROI)
	print ("Z Coord for grasping",z_coord)
	yumi_pose = cam2robot(align[0], align[1], z_coord, scene.K,'left')
	yumi_pose[2] = 0.0169 #Constant depth  
	print(yumi_pose)
	a = input("Do a depth Check ")
	approach.surgeme2(1,yumi_pose,'left')
	
	
	print("Finished Grasping")

	time.sleep(5)
	approach.surgeme3(1,'left')
	time.sleep(5)


	# cv2.imshow("Raw",scene.color_frame)
	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		exit(0)
	# rospy.spin()



# Executing LIFT

# lift = S2()
# lift.ret_to_neutral('left')
# time.sleep(3)
# t.left_close()
# time.sleep(1.5)
# lift.surgeme2(1,[0.37284002,0.07196 ,0.03],'left')



# ########################################################################################################################
# # Executing transfer approach
# approach.ret_to_neutral('left')
# time.sleep(3)
# t.right_open()
# time.sleep(2)
# t.surgeme3([0,0.0075,0])

# #########################################################################################################################
# #Executing surgeme tranfer
# transfer = S4()
# transfer.surgeme4('l-r')
# time.sleep(5)
# approach.ret_to_neutral('left')
# approach.ret_to_neutral('right')

# ###########################################################################################################################
# # Approach for drop
# time.sleep(1)
# print("Approach for Drop")
# approach_drop = S5()
# desired_pos = load_pose_by_path('/home/natalia/yumi_execution/poses/right_7_1_pose')
# print "Desired Pos: ", desired_pos['right'].translation
# approach_drop.surgeme5(1,desired_pos['right'].translation + [-0.005,-0.007,0.006],'right')

# time.sleep(1)
# t.right_open()
# approach_drop.ret_to_neutral('right')




