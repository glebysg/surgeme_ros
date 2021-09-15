import logging
import time
import os
import unittest
import numpy as np
import copy
import sys
import random
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
        self.neutral_pose = load_pose_by_path('data/neutral_pose_straight.txt')
        print('Neutral pose ', self.neutral_pose)
        self.neutral_angles = load_pose_by_path('data/neutral_joints_straight.txt')
        self.transfer_pose_high_left = load_pose_by_path('data/transfer_pose_high.txt')
        self.transfer_pose_low_left = load_pose_by_path('data/transfer_pose_low_left_to_right_new.txt')
        self.transfer_pose_high_right = load_pose_by_path('data/transfer_pos_high_right_to_left.txt')
        self.transfer_pose_low_right = load_pose_by_path('data/transfer_pose_low_right_to_left.txt')
        self.ROBOT_BUSY = False
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
        self.grasp_offset = np.array([0,0,0])

    def ret_to_neutral(self,limb):
        self.y.set_v(80)
        arm = self.y.right if limb == 'right' else self.y.left
        curr_pos_left = arm.get_pose()
        des_pos_left = curr_pos_left
        des_pos_left.translation = self.neutral_pose[limb].translation
        des_pos_left.rotation = self.neutral_pose[limb].rotation
        self.ROBOT_BUSY = True
        arm.goto_pose(des_pos_left,False,True,False)
        time.sleep(0.5)
        self.ROBOT_BUSY = False
        print "Moved to neutral :)"

    def ret_to_neutral_angles(self,limb):
        self.y.set_v(80)
        # arm = self.y.right if limb == 'right' else self.y.left
        limb_angles = 'left_angles' if limb == 'left' else 'right_angles'
        # curr_pos_limb = arm.get_state()
        # des_pos_limb = curr_pos_limb
        # des_pos_limb.joints = self.neutral_angles[limb_angles].joints
        self.ROBOT_BUSY = True
        self.goto_joint_state(self.neutral_angles[limb_angles].joints,limb)
        time.sleep(0.5)
        self.ROBOT_BUSY = False
        print "Moved to neutral :)"
        # arm.goto_state(des_pos_limb)
        print "Moved to neutral :)"

    def get_curr_pose(self,limb):
        arm = self.y.right if limb == 'right' else self.y.left
        return arm.get_pose()

    def get_curr_joints(self,limb):
        arm = self.y.right if limb == 'right' else self.y.left
        return list(arm.get_state().joints)

    def left_close(self):
        self.ROBOT_BUSY = True
        self.y.left.close_gripper(force=12,wait_for_res=False)
        time.sleep(0.1)
        self.ROBOT_BUSY = False

    def left_open(self, gripper_value=0.005):
        self.ROBOT_BUSY = True
        self.y.left.move_gripper(gripper_value)
        self.ROBOT_BUSY = False

    def right_close(self):
        self.ROBOT_BUSY = True
        self.y.right.close_gripper(force=12,wait_for_res=False)
        time.sleep(0.1)
        self.ROBOT_BUSY = False

    def right_open(self, gripper_value=0.005):
        self.ROBOT_BUSY = True
        self.y.right.move_gripper(gripper_value)
        self.ROBOT_BUSY = False

    def goto_joint_state(self,joints,limb):
        self.ROBOT_BUSY = True
        arm = self.y.right if limb == 'right' else self.y.left
        joint_state = YuMiState(vals=joints)
        try:
            arm.goto_state(joint_state)
        except:
            time.sleep(0.01)
            # try a second time
            arm.goto_state(joint_state)
        time.sleep(0.5)
        self.ROBOT_BUSY = False

    def joint_orient(self,limb,j_val,offset = 150):
        self.ROBOT_BUSY = True
        arm = self.y.right if limb == 'right' else self.y.left
        temp = arm.get_state()
        # if limb == 'left':
        #     offset = 150
        #     arm_scale = interp1d([0,180],[-30,offset])
        #     peg_scale = interp1d([0,offset],[-30,offset])
        # elif limb == 'right':
        #     offset = 50
        #     arm_scale = interp1d([0,180],[-130,offset])
        #     peg_scale = interp1d([0,offset],[-130,offset])
        curr_angles = temp.joints
        print("Curr Angles before turning joint 6:",curr_angles)
        curr_j5 = curr_angles[5]
        print ("Joint 6 degrees ",curr_j5)
        if limb =='left':
            off_angle = j_val - 40
            curr_angles[5] = off_angle
        else:
            off_angle = j_val -126
            curr_angles[5] = off_angle 
        print("Joint 6 new", curr_angles[5])
        # if (curr_j5) - (arm_scale(j_val)) > 0 :
        #     # print ("Subracing diff value ")
        #     diff_angle = (curr_j5)-(arm_scale(j_val))
        #     curr_angles[5] = curr_j5-diff_angle
        # else:
        #     diff_angle = (curr_j5)-(arm_scale(j_val))
        #     curr_angles[5] = curr_j5+abs(diff_angle)
            # print ("Adding diff value ")
        # print("Curr Angles after getting turn value for  joint 5:",curr_angles)
        # a = input("Are you satsis")
        temp.joints = curr_angles
        arm.goto_state(temp) 
        time.sleep(0.5)
        self.ROBOT_BUSY = False

    def surgeme1(self,peg,desired_pos,limb):
        self.ROBOT_BUSY = True
        self.y.set_v(80)
        arm = self.y.right if limb == 'right' else self.y.left
        if limb == "left":
            self.left_open(0.006)
        else:
            self.right_open(0.006)
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
        self.ROBOT_BUSY = False

    def surgeme2(self,peg,desired_pos,limb):
        self.ROBOT_BUSY = True
        self.y.set_v(25)
        arm = self.y.right if limb == 'right' else self.y.left
        curr_pos = arm.get_pose()

        print "Current location: ", curr_pos.translation
        print "Approaching desired peg: ",peg
        des_pos = curr_pos
        print("des pos",des_pos)
        #print "Desired_POS",desired_pos
        # desired_pos = desired_pos + self.offset_thresh
        des_pos.translation = desired_pos + self.grasp_offset
        #print "DES",des_pos_left
        arm.goto_pose(des_pos,False,True,False)
                # Block the robot only for 0.5 more seconds
        time.sleep(0.5)
        self.ROBOT_BUSY = False
                # wait for the rest of the time unblocked
        time.sleep(2.5)
                # Make the robot bussy for the post-movement status read
        self.ROBOT_BUSY = True
        curr_pos = arm.get_pose()
        self.ROBOT_BUSY = False

                # the open/close commands already block the robot
        print "Current location after moving: ", curr_pos.translation
        if limb == 'left':
            self.left_close()
        else:
            self.right_close()

    def surgeme3(self,peg,limb):#lift is hardcoded
        self.ROBOT_BUSY = True
        # self.y.set_v(80)
        arm = self.y.right if limb == 'right' else self.y.left
        arm.goto_pose_delta((0,0,0.05))
        time.sleep(1)
        self.ROBOT_BUSY = False

        # print "Shuting yumi"
        # self.y.stop()

    def surgeme4(self, limb='left'):#Tranfer Approach
        self.ROBOT_BUSY = True
        self.y.set_v(80)
        oposite_limb = 'right' if limb == 'left' else 'left'
        oposite_limb_angles = 'right_angles' if limb == 'left' else 'left_angles'
        limb_angles = 'left_angles' if limb == 'left' else 'right_angles'
        # self.y.set_v(40)
        arm = self.y.right if limb == 'right' else self.y.left
        oposite_arm = self.y.right if limb == 'left' else self.y.left
        curr_pos = arm.get_pose()
        self.ROBOT_BUSY = False
        # print "Current location after returning to neutral: ", curr_pos
        # print "Shuting yumi"
        # self.y.stop()

                # Skip the open fuction since it's blocking
        if limb == "left":
            self.right_open()
        else:
            self.left_open()

        self.ROBOT_BUSY = True
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
        time.sleep(0.1)
        self.ROBOT_BUSY = False

        ################################ DO Transer pose lose #######

        self.ROBOT_BUSY = True
        curr_pos_limb = arm.get_state()
        des_pos_limb = curr_pos_limb
        if limb == 'left':
            des_pos_limb.joints = self.transfer_pose_low_left[limb_angles].joints
            self.goto_joint_state(self.transfer_pose_low_left[limb_angles].joints,'left')
        else:
            des_pos_limb.joints = self.transfer_pose_low_right[limb_angles].joints
            self.goto_joint_state(self.transfer_pose_low_right[limb_angles].joints,'right')
        time.sleep(0.1)
        self.ROBOT_BUSY = False

#########################################################################################
        self.ROBOT_BUSY = True
        curr_pos_oposite_limb = oposite_arm.get_state()
        des_pos_oposite_limb = curr_pos_oposite_limb
        if limb == 'left':
            self.right_open(0.006) 
            time.sleep(0.5)
            des_pos_oposite_limb.joints = self.transfer_pose_low_left[oposite_limb_angles].joints
            self.goto_joint_state(self.transfer_pose_low_left[oposite_limb_angles].joints,'right')
        else:
            self.left_open(0.006) 
            time.sleep(0.5)
            des_pos_oposite_limb.joints = self.transfer_pose_low_right[oposite_limb_angles].joints
            self.goto_joint_state(self.transfer_pose_low_right[oposite_limb_angles].joints,'left')
        time.sleep(0.5)
        oposite_arm.goto_pose_delta([0.002,0.000,-0.005])
        # time.sleep(0.5)
        # oposite_arm.goto_state(des_pos_oposite_limb)
        time.sleep(0.1)
        self.ROBOT_BUSY = False

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
        arm = self.y.right if limb == 'right' else self.y.left
        opposite_arm = self.y.right if limb == 'left' else self.y.left
        self.ROBOT_BUSY = True
        self.y.set_v(80)
        if limb == 'left':
            self.right_close()
            time.sleep(0.5)
            self.left_open()
            print "Transfer Complete",limb
        else:
            self.left_close()
            time.sleep(0.5)
            self.right_open()
            print "Transfer Complete",limb  
        time.sleep(0.5)
        opposite_arm.goto_pose_delta([0,0,-0.015])
        print("WENT DOWN TO TOWN")
        time.sleep(0.5)
        self.ROBOT_BUSY = False

    def surgeme6(self,movetodelta,limb): #Approach Align and drop point 
        self.ROBOT_BUSY = True
        self.y.set_v(80)
        arm = self.y.right if limb == 'right' else self.y.left
        curr_pos = arm.get_pose()
        print("DROP POSE BEFORE EXECUTING:", curr_pos.translation)
        time.sleep(0.1)
        arm.goto_pose_delta([movetodelta[0],movetodelta[1],0])
        time.sleep(1)
        curr_pos = arm.get_pose()
        print("DROP POSE BEFORE EXECUTING:", curr_pos.translation)
        self.ROBOT_BUSY = False


    def surgeme7(self,limb): #Drop
        self.ROBOT_BUSY = True
        self.y.set_v(20)
        z=0.030
        arm = self.y.right if limb == 'right' else self.y.left
        curr_pos = arm.get_pose()
        delta_z=z-curr_pos.translation[2]
        arm.goto_pose_delta([0,0,delta_z])
        time.sleep(0.5)
        self.ROBOT_BUSY = False
        time.sleep(1)
        if limb == 'left':
            self.left_open(0.006)
        else:
            self.right_open(0.006)
        time.sleep(1.5)
        self.ROBOT_BUSY = True
        arm.goto_pose_delta([0,0,-delta_z])
        time.sleep(0.1)
        self.ROBOT_BUSY = False
        # print "Shuting yumi"
        # self.y.stop()
    
    def stop(self):
        print('Stopping Robot')
        self.y.stop()
##########################################################################################################################
