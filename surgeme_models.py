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
from scipy.interpolate import interp1d
import math


y=None

##########################################################################################################################

class Surgeme_Models():

	def __init__(self):
		self.y = YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
		global y
		y = self.y
		# self.y = y
				# TODO CHANGE THIS TO THE YUMI CLASS LATER
		self.neutral_pose = load_pose_by_path('data/neutral_pose_peg.txt')
		self.neutral_angles = load_pose_by_path('data/neutral_angles_pose_peg1.txt')
		self.transfer_pose_high_left = load_pose_by_path('data/transfer_pose_high.txt')
		self.transfer_pose_low_left = load_pose_by_path('data/transfer_pose_low.txt')
		self.transfer_pose_high_right = load_pose_by_path('data/transfer_pos_high_right_to_left.txt')
		self.transfer_pose_low_right = load_pose_by_path('data/transfer_pos_low_right_to_left.txt')

		#setup the ool distance for the surgical grippers
		# ORIGINAL GRIPPER TRANSFORM IS tcp2=RigidTransform(translation=[0, 0, 0.156], rotation=[[ 1. 0. 0.] [ 0. 1. 0.] [ 0. 0. 1.]])

		DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
		DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
		self.y.left.set_tool(DELTALEFT)
		self.y.right.set_tool(DELTARIGHT)
		self.y.set_v(80)
		self.y.set_z('z100')



		self.init_pose_left=self.y.left.get_pose()
		self.init_pose_right=self.y.right.get_pose()
		#self.offset_thresh = [0,0,random.uniform(0.005,0.01)]
		self.offset_thresh = [0,0,0]

	def ret_to_neutral(self,limb):
		self.y.set_v(80)
		arm = self.y.right if limb == 'right' else self.y.left

		curr_pos_left = arm.get_pose()
		des_pos_left = curr_pos_left
		des_pos_left.translation = self.neutral_pose[limb].translation
		des_pos_left.rotation = self.neutral_pose[limb].rotation
		arm.goto_pose(des_pos_left,False,True,False)
		print "Moved to neutral :)"


	def ret_to_neutral_angles(self,limb):
		self.y.set_v(80)
		arm = self.y.right if limb == 'right' else self.y.left

		limb_angles = 'left_angles' if limb == 'left' else 'right_angles'

		curr_pos_limb = arm.get_state()
		des_pos_limb = curr_pos_limb
		des_pos_limb.joints = self.neutral_angles[limb_angles].joints
		arm.goto_state(des_pos_limb)
		print "Moved to neutral :)"
	
	def get_curr_pose(self,limb):
		arm = self.y.right if limb == 'right' else self.y.left
		return arm.get_pose()
	
	def left_close(self):
		self.y.left.close_gripper(force=12,wait_for_res=False)
	
	def left_open(self):
		self.y.left.move_gripper(0.005)
	
	def right_close(self):
		self.y.right.close_gripper(force=12,wait_for_res=False)

	def right_open(self):
		self.y.right.move_gripper(0.005)

	def joint_orient(self,limb,j_val,offset = 150):
	 	

		
		arm = self.y.right if limb == 'right' else self.y.left
		temp = arm.get_state()
		if limb == 'left':
			offset = 150
			arm_scale = interp1d([0,180],[-30,offset]) 
			peg_scale = interp1d([0,offset],[-30,offset])
		elif limb == 'right':
			offset = 50
			arm_scale = interp1d([0,180],[-130,offset]) 
			peg_scale = interp1d([0,offset],[-130,offset])
		curr_angles = temp.joints
		# print("Curr Angles before turning joint 5:",curr_angles)
		curr_j5 = curr_angles[5]
		print ("Joint 5 degrees ",curr_j5)
		if (curr_j5) - (arm_scale(j_val)) > 0 :
			# print ("Subracing diff value ")
			diff_angle = (curr_j5)-(arm_scale(j_val))
			curr_angles[5] = curr_j5-diff_angle
		else:
			diff_angle = (curr_j5)-(arm_scale(j_val))
			curr_angles[5] = curr_j5+abs(diff_angle)
			# print ("Adding diff value ")
		# print("Curr Angles after getting turn value for  joint 5:",curr_angles)
		# a = input("Are you satsis")
		temp.joints = curr_angles
		arm.goto_state(temp) 
		time.sleep(0.5)

	def surgeme1(self,peg,desired_pos,limb):
		self.y.set_v(80)
		arm = self.y.right if limb == 'right' else self.y.left
		curr_pos = arm.get_pose()
		print "Current location: ", curr_pos.translation
		# print "Approaching desired peg: ",peg
		des_pos = curr_pos
		#print "Desired_POS",desired_pos
		desired_pos = desired_pos + self.offset_thresh
		des_pos.translation = desired_pos
		#print "DES",des_pos_left
		arm.goto_pose(des_pos,False,True,False)
		time.sleep(0.2)

		curr_pos = arm.get_pose()
		print "Current location after moving: ", curr_pos.translation
		
		

	def surgeme2(self,peg,desired_pos,limb):
		self.y.set_v(25)	
		arm = self.y.right if limb == 'right' else self.y.left
		curr_pos = arm.get_pose()

		print "Current location: ", curr_pos.translation
		print "Approaching desired peg: ",peg
		des_pos = curr_pos
		print("des pos",des_pos)
		#print "Desired_POS",desired_pos
		# desired_pos = desired_pos + self.offset_thresh
		des_pos.translation = desired_pos
		#print "DES",des_pos_left
		arm.goto_pose(des_pos,False,True,False)
		time.sleep(3)

		curr_pos = arm.get_pose()
		print "Current location after moving: ", curr_pos.translation
		if limb == 'left':
			self.left_close()
		else:
			self.right_close()


	def surgeme3(self,peg,limb):#lift is hardcoded
		self.y.set_v(80)
		arm = self.y.right if limb == 'right' else self.y.left
		curr_pos = arm.get_pose()
		print "Current location: ", curr_pos.translation
		print "Approaching desired peg: ",peg
		des_pos = curr_pos
		#print "Desired_POS",desired_pos
		desired_pos = des_pos.translation
		desired_pos[2] = 0.05 
		des_pos.translation = desired_pos
		#print "DES",des_pos_left
		arm.goto_pose(des_pos,False,True,False)
		time.sleep(1)

		# print "Shuting yumi"
		# self.y.stop()
		


	def surgeme4(self, limb='left'):#Tranfer Approach
		self.y.set_v(80)
		oposite_limb = 'right' if limb == 'left' else 'left'
		oposite_limb_angles = 'right_angles' if limb == 'left' else 'left_angles'
		limb_angles = 'left_angles' if limb == 'left' else 'right_angles'
		# self.y.set_v(40)
		arm = self.y.right if limb == 'right' else self.y.left
		oposite_arm = self.y.right if limb == 'left' else self.y.left
		curr_pos = arm.get_pose()
		# print "Current location after returning to neutral: ", curr_pos
		# print "Shuting yumi"
		# self.y.stop()
		oposite_arm.move_gripper(0.006)
		curr_pos_limb = arm.get_state()
		des_pos_limb = curr_pos_limb 
		if limb == 'left':
			des_pos_limb.joints = self.transfer_pose_high_left[limb_angles].joints
		else:
			des_pos_limb.joints = self.transfer_pose_high_right[limb_angles].joints
		arm.goto_state(des_pos_limb)

		curr_pos_oposite_limb = oposite_arm.get_state()
		des_pos_oposite_limb = curr_pos_oposite_limb

		if limb == 'left':
			des_pos_oposite_limb.joints = self.transfer_pose_high_left[oposite_limb_angles].joints
		else:
			des_pos_oposite_limb.joints = self.transfer_pose_high_right[oposite_limb_angles].joints

		oposite_arm.goto_state(des_pos_oposite_limb)

		time.sleep(1)
		################################ DO Transer pose lose #######

		curr_pos_limb = arm.get_state()
		des_pos_limb = curr_pos_limb 
		if limb == 'left':
			des_pos_limb.joints = self.transfer_pose_low_left[limb_angles].joints
		else:
			des_pos_limb.joints = self.transfer_pose_low_right[limb_angles].joints
		arm.goto_state(des_pos_limb)

		curr_pos_oposite_limb = oposite_arm.get_state()
		des_pos_oposite_limb = curr_pos_oposite_limb
		if limb == 'left':
			des_pos_oposite_limb.joints = self.transfer_pose_low_left[oposite_limb_angles].joints
		else:
			des_pos_oposite_limb.joints = self.transfer_pose_low_right[oposite_limb_angles].joints

		oposite_arm.goto_state(des_pos_oposite_limb)

		time.sleep(1)
		# print "Shuting yumi"
		# self.y.stop()
		
		# Go above the transfer pose
		# curr_pos_limb = arm.get_pose()
		# des_pos_limb = curr_pos_limb 
		# des_pos_limb.translation = self.transfer_pose_high[limb].translation
		# # des_pos_limb.rotation = self.transfer_pose_high[limb].rotation

		# curr_pos_oposite_limb = oposite_arm.get_pose()
		# des_pos_oposite_limb = curr_pos_oposite_limb
		# des_pos_oposite_limb.translation = self.transfer_pose_high[oposite_limb].translation
		# # des_pos_oposite_limb.rotation = self.transfer_pose_high[oposite_limb].rotation
		# # print "Moved to Transfer Approach "
		# #self.y.goto_pose_sync(des_pos_left, des_pos_right)
		# arm.goto_pose(des_pos_limb,False,True,False)
		# arm.goto_pose(des_pos_oposite_limb,False,True,False)
		# time.sleep(7)
		# # Tranfer pose
		# curr_pos_limb = arm.get_pose()
		# des_pos_limb = curr_pos_limb 
		# des_pos_limb.translation = self.transfer_pose_low[limb].translation
		# # des_pos_limb.rotation = self.transfer_pose_low[limb].rotation

		# curr_pos_oposite_limb = oposite_arm.get_pose()
		# des_pos_oposite_limb = curr_pos_oposite_limb
		# des_pos_oposite_limb.translation = self.transfer_pose_low[oposite_limb].translation
		# # des_pos_oposite_limb.rotation = self.transfer_pose_low[oposite_limb].rotation
		# print "Moved to Transfer Approach "
		# #self.y.goto_pose_sync(des_pos_left, des_pos_right)
		# arm.goto_pose(des_pos_limb,False,True,False)
		# arm.goto_pose(des_pos_oposite_limb,False,True,False)
		# time.sleep(3)

	def surgeme5(self,limb='left'):
		self.y.set_v(80)
		if limb == 'left':
			self.right_open()
			time.sleep(0.5)
			self.right_close()
			time.sleep(0.5)
			self.left_open()
			print "Transfer Complete",limb
		else:
			self.left_open()
			time.sleep(0.5)
			self.left_close()
			time.sleep(0.5)
			self.right_open()
			print "Transfer Complete",limb	


	def surgeme6(self,movetodelta,limb): #Approach Align and drop point 
		self.y.set_v(80)
		arm = self.y.right if limb == 'left' else self.y.left
		arm.goto_pose_delta([movetodelta[0],movetodelta[1],0])
		time.sleep(1)


	def surgeme7(self,limb): #Drop
		self.y.set_v(20)
		z=0.025
		arm = self.y.right if limb == 'left' else self.y.left
		curr_pos = arm.get_pose()
		delta_z=z-curr_pos.translation[2]
		arm.goto_pose_delta([0,0,delta_z])
		time.sleep(1.5)
		arm.move_gripper(0.007)
		time.sleep(1.5)
		arm.goto_pose_delta([0,0,-delta_z])
		# print "Shuting yumi"
		# self.y.stop()
##########################################################################################################################
