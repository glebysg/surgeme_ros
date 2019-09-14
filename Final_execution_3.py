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

	def depth_cb(self,data):
	
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
		self.pegs = pegs #here saved ROIS of triangles
		# print(self.pegs)
		# [[149, 211, 171, 231], [225, 289, 338, 406], [143, 208, 331, 383]]

		# mask_img = data.masks
		# print(mask_img.encoding)
		if len(data.masks)>0:
			if len(data.masks[0].data)>0:
				mask_frame = self.bridge.imgmsg_to_cv2(data.masks[0],'passthrough')
				self.mask_frame = cv2.cvtColor(mask_frame,cv2.COLOR_BGR2RGB)
			# print self.mask_frame.shape
		# cv2.imshow("MASK",self.mask_frame)
		# cv2.waitKey(0)

	def pole_cb(self,data):
		poles = []
		count = 0
		if len(data.bounding_boxes)>=12:
			for pole in data.bounding_boxes:
				if pole.Class == "pole"+str(count):

					poles.append([(pole.xmin+pole.xmax)/2,(pole.ymin+pole.ymax)/2])
					count = count+1
			count = 0
			self.poles_found = np.array(poles)
			# print self.poles_found
			self.pole_flag = 1

########################### ENd of all callback functions #########################

	def grasp_points(self,ROI,bbox,limb, offset=0):

		gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)
		# dst = cv2.cornerHarris(gray,5,9,0.19)
		# dst = cv2.dilate(dst,None)
		# print dst.shape
		h,w = gray.shape
		h = h/2
		w = w/2
		cor_length = 0
		# cv2.imshow("View_ROI",gray)
		# cv2.waitKey(0)
		while cor_length<3:
			print("Finding Corners")
			cor = cv2.goodFeaturesToTrack(gray,3,0.01,48)
			# print(cor)
			cor = np.int0(cor)
			corners = []
			# print(cor)
	        
			for i in cor:
				x,y = i.ravel()
				corners.append([x,y])
				cv2.circle(ROI,(x,y),3,255,-1)

			corners = np.array(corners)
			cor_length = len(corners)
			print(cor_length)

		pole = get_center_triangle(corners)
		
		gpoints = []
		corner_pairs = np.array([[[0,0],[0,0]]])
		# Get the actual grasping points, inside the object of interest
		for point1, point2 in list(itertools.combinations(corners, 2)):
			
			center = np.array([abs(point1[0]+point2[0])/2, abs(point1[1]+point2[1])/2])
			corner_pairs = np.concatenate((corner_pairs,np.array([[point1,point2]])),axis = 0)

			# if limb == 'left':
			gpoint_plus = center + normal_to_line(point1,point2)*offset
			gpoint_minus = center - normal_to_line(point1,point2)*offset
			# else:
			# 	gpoint_plus = center + normal_to_line(point1,point2)*-offset
			# 	gpoint_minus = center - normal_to_line(point1,point2)*-offset
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
		if limb == 'left':
			grasp_point,grasp_index,corner_point,corner_index = min_dist_robot2points([0,w],gpoints,corners)#h,0 for left hand or h and w*2 for right hand
		else:
			grasp_point,grasp_index,corner_point,corner_index = min_dist_robot2points([h*2,w],gpoints,corners)#h,0 for left hand or h and w*2 for right hand
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

		opposite_limb = 'right' if limb == 'left' else 'left'
		gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)

		h,w = gray.shape
		cor = cv2.goodFeaturesToTrack(gray,10,0.005,15)
		cor = np.int0(cor)
		corners = []
		for i in cor:
			x,y = i.ravel()
			corners.append([x,y])
			# cv2.circle(ROI,(x,y),3,255,-1)

		corners = np.array(corners)
		#corner_point is the opposite corner point in pixels
		if limb == 'left':
			corner_point,corner_index = min_dist_robot2drop_pt([0,w/2],corners)#h,0 for left hand or h and w*2 for right hand
		elif limb == 'right':
			corner_point,corner_index = min_dist_robot2drop_pt([h,w/2],corners)#h,0 for left hand or h and w*2 for right hand
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

		corner_robot_pose = cam2robot(cpoint[0], cpoint[1], z_coord, scene.K,opposite_limb)# we take oposite limb because this is drop limb 

		center_coord_x = (rob_pose.translation[0])+(corner_robot_pose[0]-rob_pose.translation[0])/3
		center_coord_y = (rob_pose.translation[1])+(corner_robot_pose[1]-rob_pose.translation[1])/3

		print("gripper pose",rob_pose.translation[0],rob_pose.translation[1])
		print("corner pose",corner_robot_pose[0],corner_robot_pose[1])

		pole_pose = [pole_pos[0],pole_pos[1]]
		movetodelta=[pole_pose[0]-center_coord_x,pole_pose[1]-center_coord_y]
		print ("it will move to delta",movetodelta)
		cv2.imshow("Drop_Corners",ROI)
		cv2.waitKey(0)

		return movetodelta

	def pole_positions_rob(self,pole_poses,robot):

		if robot=="yumi":
			left_ids=[0,5,7,10,6,2]
			right_ids = [1,4,9,11,8,3]
		if robot=="taurus":
			right_ids=[0,5,7,10,6,2]
			left_ids = [1,4,9,11,8,3]
		# left_ids_vishnu = [0,2,3,6,7,10]

		poles_left = []
		# right_ids_vishnu = [1,4,5,8,9,11]

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
		self.left_poles_pixel_location=poles_left
		# print("poles left pos pix",self.left_poles_pixel_location)
		poles_left_arm = cam2robot_array(poles_left,self.K,limb)

		for i in right_ids:
			poles_right.append(poles_xyz[i])
		poles_right = np.array(poles_right)
		self.right_poles_pixel_location=poles_right
		print("poles right pos pix",self.right_poles_pixel_location)
		poles_right_arm = cam2robot_array(poles_right,self.K,opposite_limb)

		cv2.imshow("Poles",self.color_frame)
		cv2.waitKey(0)


		return poles_left_arm,poles_right_arm

	def detect_ROI(self,arm,selected_pole):
		triangles=self.pegs #triangles detected in order of confidence
		
		if arm=="left":
			poles_cam=self.left_poles_pixel_location
		if arm=="right":
			poles_cam=self.right_poles_pixel_location

		print("poles location shape",poles_cam.shape)
		selected_pole_XY=poles_cam[selected_pole-1,0:2]#selected pole is from 1-6, so -1 for 0-5

		triangle_ids=range(len(triangles))
		closest_id=0
		dist=1000000000

		for i in triangle_ids:
			bbox=np.array(triangles[i])
			center_dist_2_pole=[(bbox[1]+bbox[0])/2-selected_pole_XY[0],(bbox[2]+bbox[3])/2-selected_pole_XY[1]]
			abs_dist=math.sqrt(center_dist_2_pole[0]**2+center_dist_2_pole[1]**2)
			if abs_dist<dist:
				dist=abs_dist
				closest_id=i
		print("dist",dist,"triangle id",closest_id)
		return(closest_id)

	def subscribe(self):
		rospy.init_node('BoundingBoxes', anonymous=True)
		# rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.get_bbox) #This is tthe subscriber for the darknet bounding boxes. SInce we use mask we dont need this
		rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb)
		rospy.Subscriber("/camera/color/camera_info",CameraInfo, self.camera_callback)
		rospy.Subscriber("/camera/color/image_raw",Image, self.image_callback)
		rospy.Subscriber("/masks_t",ObjMasks,self.mask_callback)
		rospy.Subscriber("/darknet_ros/tracked_bbs",BoundingBoxes,self.pole_cb)

##########################################################################################################################
########################################## Surgeme Definitions ##############################

def S1(calign,z_coord,g_angle,K,limb):

	execution.joint_orient(limb,g_angle)
	yumi_pose = cam2robot(calign[0], calign[1], z_coord , K,limb)
	yumi_pose[2] = 0.07
	execution.surgeme1(1,yumi_pose,limb)
	# time.sleep(5)
	print("Finished Approach")

def S2(align,z_coord,K,limb):
	yumi_pose = cam2robot(align[0], align[1], z_coord,K,limb)
	yumi_pose[2] = 0.0179 #Constant depth  

	execution.surgeme2(1,yumi_pose,limb)
	print("Finished Grasping")

def S3(limb):
	execution.surgeme3(1,limb)
	print("FInished Lift")
	execution.ret_to_neutral_angles(limb)


def S4(limb):
	execution.surgeme4(limb)
	print ("Finished Go TO Transfer ")

def S5(limb):
	execution.surgeme5(limb)
	print("Finsihed Transfer ")
	execution.ret_to_neutral_angles(limb)
	execution.ret_to_neutral_angles(opposite_limb)
	time.sleep(5)
	transfer_flag = 1
	return transfer_flag


def S6(ROI,first_peg,drop_pose,limb):
	drop_pt = scene.drop_pose(ROI,limb,first_peg,drop_pose)
	execution.surgeme6(drop_pt,limb)
	print ("Finished Approach ")


def S7(limb,opposite_limb):
	execution.surgeme7(limb)
	print("Finished align and Drop") 
	execution.ret_to_neutral_angles(opposite_limb)



if __name__ == '__main__':

	# Start Robot Environment
	# Start surgemes class and open grippers and take robot to neutral positions
	execution = Surgemes()
	time.sleep(1)
	# limb = input('Ente the limb : ')
	limb = 'left'
	opposite_limb = 'right' if limb == 'left' else 'left'

	# drop_pole_pose = input('Enter the destination pole : ') #please refer format

	drop_pole_pose = 1
	selected_pole=int(1) #start in 1


	
	rob_pose = execution.get_curr_pose(limb)
	# execution.left_open()
	# execution.right_open()
	# time.sleep(2)
	execution.ret_to_neutral_angles(limb)
	execution.ret_to_neutral_angles(opposite_limb)
	# time.sleep(5)

	#################################### Initial setup complete 

	ROI_offset = 10 #ROI bounding box offsets
	# Start the Scene calss to obtain pole positions and bounding boxes etc. 
	scene = Scene()
	scene.subscribe()
	time.sleep(2)


	while scene.pole_flag == 0:
		a = 1
	# print(scene.poles_found)

	left_poles,right_poles = scene.pole_positions_rob(scene.poles_found,"yumi") #Pole positions in robot space
	if opposite_limb == 'right':
		drop_pole_pose = right_poles[drop_pole_pose]
	else:
		drop_pole_pose = left_poles[drop_pole_pose]

	stop = 0
	surgeme_no = 0
	while(surgeme_no < 8):
		surgeme_no = input('Enter the surgeme number youd like to perform: ')
		surgeme_no = int(surgeme_no)
		

		if surgeme_no == 1:
			while len(scene.pegs) == 0:#wait for pegs to be deteced
				# print("Enter")
				a = 1

			selected_triangle_id=scene.detect_ROI(limb,selected_pole)
			# selected_triangle_id=1
			print(selected_triangle_id)
			print("esto",scene.pegs[int(selected_triangle_id)])
			first_peg = np.array(scene.pegs[int(selected_triangle_id)]) #xmin,xmax,ymin,ymax choose peg closest to pole required 
			print("first peg found",first_peg)
			first_peg = first_peg.reshape(4)
			# Apply offsets to pegs
			first_peg[0] = first_peg[0]-ROI_offset
			first_peg[2] = first_peg[2]-ROI_offset
			first_peg[1] = first_peg[1]+ROI_offset
			first_peg[3] = first_peg[3]+ROI_offset
			# print first_peg

			while len(scene.mask_frame) == 0:
				# print("Entered")
				a = 1	

			ROI = scene.mask_frame[first_peg[0]:first_peg[1],first_peg[2]:first_peg[3],:] #xmin,xmax,ymin,ymax
			depth_ROI = scene.depth_vals[first_peg[0]:first_peg[1],first_peg[2]:first_peg[3]]
			grasp,g_angle,corner = scene.grasp_points(ROI,first_peg,limb)# obtain corner points and grasp poiints from scene 
			
		
		if surgeme_no == 1:
			
			S1(corner,get_max_depth(depth_ROI),g_angle,scene.K,limb)#Perform Approach

		if surgeme_no == 2:
			S2(grasp,get_min_depth(depth_ROI),scene.K,limb)#Perform Grasp
		
		if surgeme_no == 3:
			S3(limb)#Perform Lift
		if surgeme_no == 4:
			S4(limb)#Perform Go To transfer
		if surgeme_no == 5:
			transfer_flag = S5(limb)


		if surgeme_no == 6:

			while len(scene.pegs) == 0:
				a = 1
			first_peg = []
			first_peg = np.array(scene.pegs) #Choose peg closest to opposite limb gripper
			first_peg = first_peg.reshape(4)
			first_peg[0] = first_peg[0]-ROI_offset
			first_peg[2] = first_peg[2]-ROI_offset
			first_peg[1] = first_peg[1]+ROI_offset
			first_peg[3] = first_peg[3]+ROI_offset

			while len(scene.mask_frame) == 0:
				a = 1	

			transfer_flag = 0
			ROI = scene.mask_frame[first_peg[0]:first_peg[1],first_peg[2]:first_peg[3],:] #xmin,xmax,ymin,ymax
			
		if surgeme_no == 6: 
			S6(ROI,first_peg,drop_pole_pose,limb)#Perform Approach
		
		if surgeme_no == 7:
			S7(limb,opposite_limb)#Perform Drop
		
		# stop = input('Do you want to stop: ')
		# stop = int(stop)
	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		exit(0)
	rospy.spin()


