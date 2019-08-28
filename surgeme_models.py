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
from helpers import load_pose_by_path

##########################################################################################################################
#Approach
class S1:

	def __init__(self):
		self.y = YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)

				# TODO CHANGE THIS TO THE YUMI CLASS LATER
		self.neutral_pose = load_pose_by_path('data/yumi_default_peg.txt')
		#setup the ool distance for the surgical grippers
		# ORIGINAL GRIPPER TRANSFORM IS tcp2=RigidTransform(translation=[0, 0, 0.156], rotation=[[ 1. 0. 0.] [ 0. 1. 0.] [ 0. 0. 1.]])

		DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
		DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
		self.y.left.set_tool(DELTALEFT)
		self.y.right.set_tool(DELTARIGHT)
		self.y.set_v(40)
		self.y.set_z('z100')



		self.init_pose_left=self.y.left.get_pose()
		self.init_pose_right=self.y.right.get_pose()
		#self.offset_thresh = [0,0,random.uniform(0.005,0.01)]
		self.offset_thresh = [0,0,0]

	def ret_to_neutral(self,limb):
		arm = self.y.right if limb == 'right' else self.y.left

		curr_pos_left = arm.get_pose()
		des_pos_left = curr_pos_left
		des_pos_left.translation = self.neutral_pose[limb].translation
		des_pos_left.rotation = self.neutral_pose[limb].rotation
		arm.goto_pose(des_pos_left,False,True,False)
		print "Moved to neutral :)"



	def surgeme1(self,peg,desired_pos,limb):
		arm = self.y.right if limb == 'right' else self.y.left
		curr_pos = arm.get_pose()
		print "Current location: ", curr_pos.translation
		print "Approaching desired peg: ",peg
		des_pos = curr_pos
		#print "Desired_POS",desired_pos
		desired_pos = desired_pos + self.offset_thresh
		des_pos.translation = desired_pos
		#print "DES",des_pos_left
		arm.goto_pose(des_pos,False,True,False)
		time.sleep(5)

		curr_pos = arm.get_pose()
		print "Current location after moving: ", curr_pos.translation

#########################################################################################################################
#Lift
class S2:

	def __init__(self):

		# self.y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
		global y
		self.y = y
		#setup the ool distance for the surgical grippers
		# ORIGINAL GRIPPER TRANSFORM IS tcp2=RigidTransform(translation=[0, 0, 0.156], rotation=[[ 1. 0. 0.] [ 0. 1. 0.] [ 0. 0. 1.]])

		DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
		DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
		self.y.left.set_tool(DELTALEFT)
		self.y.right.set_tool(DELTARIGHT)
		self.y.set_v(40)
		self.y.set_z('z100')



		self.init_pose_left=self.y.left.get_pose()
		self.init_pose_right=self.y.right.get_pose()
		
		
	def ret_to_neutral(self,limb):



		if limb == 'left':# assigned peg value of 1 
			curr_pos_left = self.y.left.get_pose()
			des_pos_left = curr_pos_left
			des_pos_left.translation = [0.37278003,0.0719 -0.007,0.00775+0.001]
			des_pos_left.rotation = [[-0.47578654, -0.87380913 , 0.10042293],[-0.75995407,  0.35091608, -0.54710851], [ 0.44282839, -0.33662368, -0.83101595]]

			self.y.left.goto_pose(des_pos_left,False,True,False)
			#time.sleep(5)
			print "Moved to neutral "



		if limb == 'right': # Need to assign peg values
			curr_pos_right = self.y.right.get_pose()
			des_pos_right = curr_pos_right
			des_pos_right.translation = [ 0.37278003, -0.08536001 , 0.00775]
			des_pos_right.rotation = [[-0.2283057  , 0.88719064 , 0.40096044], [ 0.69851396, -0.13761998 , 0.70223855], [ 0.67819964 , 0.44040153 ,-0.58829562]]
			self.y.right.goto_pose(des_pos_right,False,True,False)
			print "Moved to neutral "

		if limb == 'both':
			curr_pos_left = self.y.left.get_pose()
			des_pos_left = curr_pos_left
			des_pos_left.translation = [0.33670002,0.11435001,0.10492]
			des_pos_left.rotation = [[-0.39892747,-0.82643495 ,0.39731869], [-0.72430491, 0.01826895 ,-0.68923773], [ 0.56235155 ,-0.56273574 ,-0.60587888]]

			curr_pos_right = self.y.right.get_pose()
			des_pos_right = curr_pos_right
			des_pos_right.translation = [ 0.33832002, -0.08536001 , 0.10328]
			des_pos_right.rotation = [[-0.2283057  , 0.88719064 , 0.40096044], [ 0.69851396, -0.13761998 , 0.70223855], [ 0.67819964 , 0.44040153 ,-0.58829562]]
			print "Moved to neutral "
			self.y.goto_pose_sync(des_pos_left, des_pos_right)


	def surgeme2(self,peg,desired_pos ,limb):

		if limb == 'left':
			curr_pos_left = self.y.left.get_pose()
			print "Current location: ", curr_pos_left.translation
			print "Approaching desired peg: ",peg 
			des_pos_left = curr_pos_left
			

			des_pos_left.translation = desired_pos
			
			self.y.left.goto_pose(des_pos_left,False,True,False)
			time.sleep(5)

			curr_pos_left = self.y.left.get_pose()
			print "Current location after moving: ", curr_pos_left.translation



		if limb == 'right':
			curr_pos_right = self.y.right.get_pose()
			print "Current location: ", curr_pos_right.translation
			print "Approaching desired peg: ",peg 
			des_pos_right = curr_pos_right
			des_pos_right.translation = desired_pos
			self.y.right.goto_pose(des_pos_right,False,True,False)
			time.sleep(5)
			print "Current location after moving: ", self.y.right.get_pose().translation


#########################################################################################################################
#Transfer-Approach

class S3:

	def __init__(self):

		# self.y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
		global y
		self.y = y
		#setup the ool distance for the surgical grippers
		# ORIGINAL GRIPPER TRANSFORM IS tcp2=RigidTransform(translation=[0, 0, 0.156], rotation=[[ 1. 0. 0.] [ 0. 1. 0.] [ 0. 0. 1.]])

		DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
		DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
		self.y.left.set_tool(DELTALEFT)
		self.y.right.set_tool(DELTARIGHT)
		self.y.set_v(40)
		self.y.set_z('z100')



		self.init_pose_left=self.y.left.get_pose()
		self.init_pose_right=self.y.right.get_pose()
		self.offset = 0.0040
		
		
	def left_close(self):
		self.y.left.close_gripper(force=2,wait_for_res=False)
	
	def left_open(self):
		self.y.left.move_gripper(0.005)
	
	def right_close(self):
		self.y.right.close_gripper(force=2,wait_for_res=False)

	def right_open(self):
		self.y.right.move_gripper(0.005)


	


	def surgeme3(self,desired_pos):

		curr_pos_left = self.y.left.get_pose()
		des_pos_left = curr_pos_left 
		des_pos_left.translation[1] = desired_pos[1] 

		curr_pos_right = self.y.right.get_pose()
		des_pos_right = curr_pos_right
		des_pos_right.translation[1] = desired_pos[1] -  self.offset
		print "Moved to Transfer Approach "
		#self.y.goto_pose_sync(des_pos_left, des_pos_right)
		self.y.right.goto_pose(des_pos_right,False,True,False)
		self.y.left.goto_pose(des_pos_left,False,True,False)



#########################################################################################################################
# Transfer - Transfer
class S4:

	def __init__(self):

		# self.y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
		global y
		self.y = y
		#setup the ool distance for the surgical grippers
		# ORIGINAL GRIPPER TRANSFORM IS tcp2=RigidTransform(translation=[0, 0, 0.156], rotation=[[ 1. 0. 0.] [ 0. 1. 0.] [ 0. 0. 1.]])

		DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
		DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
		self.y.left.set_tool(DELTALEFT)
		self.y.right.set_tool(DELTARIGHT)
		self.y.set_v(40)
		self.y.set_z('z100')



		self.init_pose_left=self.y.left.get_pose()
		self.init_pose_right=self.y.right.get_pose()
		self.offset = 0.1075
		
		
	def left_close(self):
		self.y.left.close_gripper(force=2,wait_for_res=False)
	
	def left_open(self):
		self.y.left.move_gripper(0.005)
	
	def right_close(self):
		self.y.right.close_gripper(force=2,wait_for_res=False)

	def right_open(self):
		self.y.right.move_gripper(0.005)


	


	def surgeme4(self,trans):

		if trans == 'l-r':
			self.right_open()
			time.sleep(2)
			self.right_close()
			time.sleep(2)
			self.left_open()
			print "Transfer Complete",trans


		if trans == 'r-l':
			self.left_open()
			time.sleep(2)
			self.left_close()
			time.sleep(2)
			self.right_open()
			print "Transfer Complete",trans			


#########################################################################################################################
#Approach for Drop
class S5:

	def __init__(self):

		global y
		self.y = y
		#setup the ool distance for the surgical grippers
		# ORIGINAL GRIPPER TRANSFORM IS tcp2=RigidTransform(translation=[0, 0, 0.156], rotation=[[ 1. 0. 0.] [ 0. 1. 0.] [ 0. 0. 1.]])

		DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
		DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
		self.y.left.set_tool(DELTALEFT)
		self.y.right.set_tool(DELTARIGHT)
		self.y.set_v(40)
		self.y.set_z('z100')



		self.init_pose_left=self.y.left.get_pose()
		self.init_pose_right=self.y.right.get_pose()
		#self.offset_thresh = [0,0,random.uniform(0.005,0.01)]
		self.offset_thresh = [0,0,0]  
		
	def ret_to_neutral(self,limb):

		if limb == 'left':
			curr_pos_left = self.y.left.get_pose()
			des_pos_left = curr_pos_left
			des_pos_left.translation = [0.33670002,0.11435001,0.10492]
						# des_pos_left.translation = [0.33670002,0.109,0.10492]
			des_pos_left.rotation = [[-0.39892747,-0.82643495 ,0.39731869], [-0.72430491, 0.01826895 ,-0.68923773], [ 0.56235155 ,-0.56273574 ,-0.60587888]]

			self.y.left.goto_pose(des_pos_left,False,True,False)
			#time.sleep(5)
			print "Moved to neutral "



		if limb == 'right':
			curr_pos_right = self.y.right.get_pose()
			des_pos_right = curr_pos_right
			des_pos_right.translation = [ 0.34570002, -0.08536001 , 0.1135]
			des_pos_right.rotation = [[-0.2283057  , 0.88719064 , 0.40096044], [ 0.69851396, -0.13761998 , 0.70223855], [ 0.67819964 , 0.44040153 ,-0.58829562]]
			self.y.right.goto_pose(des_pos_right,False,True,False)
			print "Moved to neutral "

		if limb == 'both':#not working 
			curr_pos_left = self.y.left.get_pose()
			des_pos_left = curr_pos_left
			des_pos_left.translation = [0.33670002,0.11435001,0.10492]
			des_pos_left.rotation = [[-0.39892747,-0.82643495 ,0.39731869], [-0.72430491, 0.01826895 ,-0.68923773], [ 0.56235155 ,-0.56273574 ,-0.60587888]]

			curr_pos_right = self.y.right.get_pose()
			des_pos_right = curr_pos_right
			des_pos_right.translation = [ 0.33832002, -0.08536001 , 0.10328]
			des_pos_right.rotation = [[-0.2283057  , 0.88719064 , 0.40096044], [ 0.69851396, -0.13761998 , 0.70223855], [ 0.67819964 , 0.44040153 ,-0.58829562]]
			print "Moved to neutral "
			self.y.goto_pose_sync(des_pos_left, des_pos_right)


	def surgeme5(self,peg,desired_pos,limb):

		if limb == 'left':
			curr_pos_left = self.y.left.get_pose()
			print "Current location: ", curr_pos_left.translation
			print "Approaching desired peg: ",peg 
			des_pos_left = curr_pos_left
			#print "Desired_POS",desired_pos
			desired_pos = desired_pos + self.offset_thresh
			des_pos_left.translation = desired_pos
			#print "DES",des_pos_left
			self.y.left.goto_pose(des_pos_left,False,True,False)
			time.sleep(5)

			curr_pos_left = self.y.left.get_pose()
			print "Current location after moving: ", curr_pos_left.translation



		if limb == 'right':
			curr_pos_right = self.y.right.get_pose()
			print "Current location: ", curr_pos_right.translation
			print "Approaching desired peg: ",peg 
			des_pos_right = curr_pos_right
			desired_pos = desired_pos + self.offset_thresh
			des_pos_right.translation = desired_pos
			self.y.right.goto_pose(des_pos_right,False,True,False)
			time.sleep(5)
			print "Current location after moving: ", self.y.right.get_pose().translation
