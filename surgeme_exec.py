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
from std_msgs.msg import String
from surgeme_models import S1
from helpers import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs
from yumi_homography_functions import *
from sensor_msgs.msg import CameraInfo


class Scene():
	def __init__(self):
		self.pegs = []
		self.bridge = CvBridge()
		self.KI = []


	def get_bbox(self,data):
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
		depth_vals = cv_image/1000


	def camera_callback(self, data):

		# print(data.K)
		# data.encoding = "bgr8"
		self.K = np.array(list(data.K)).reshape(3,3)

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

	def listener(self):
		rospy.init_node('BoundingBoxes', anonymous=True)
		rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.get_bbox)
		rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb) 
		rospy.Subscriber("/camera/color/camera_info",CameraInfo, self.camera_callback)

##########################################################################################################################


if __name__ == '__main__':
	# t = S3()
	# t.left_open()
	# t.right_open()
	scene = Scene()
	scene.listener()
	time.sleep(2)
	# rospy.spin()

	approach = S1()
	approach.ret_to_neutral('left')
	approach.ret_to_neutral('right')
	time.sleep(2)
	print "Arms at neutral"

	# Wait until a peg is found
	while len(scene.pegs) == 0:
		sleep(0.1)
	# Get the camera coords of the object of interest
	first_peg = scene.pegs[0]
	#cam_points = [first_peg[0], first_peg[2], 0.3338,1]
	cam_points = get_3dpt_depth([first_peg[0], first_peg[2]], 0.3338, scene.K)
	print("cam_points", cam_points)
	cam_points = np.concatenate((cam_points,[1]))
	world_points = camera_to_world(cam_points)
	# print("cam_points", cam_points)
	print("world_points", world_points)
	print("intrin", scene.K[0,2],scene.K[1,2],scene.K[0,0],scene.K[1,1])

	world_points = world_points
	# world_points = np.transpose(world_points)
	# print(world_points.shape)
	yumi_pose = world_to_yumi(world_points, 'left')
	
	yumi_pose = yumi_pose.reshape(3)
	print ("robot points",yumi_pose)
	approach.surgeme1(1,yumi_pose,'left')
	time.sleep(2)
	print("Finished Approach")
	rospy.spin()
	# while not rospy.is_shutdown():
	# 	time.sleep(5)
	# 	break

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




