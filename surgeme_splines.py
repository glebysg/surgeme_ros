import numpy as np
import csv
import scipy
import pickle as pkl
from helpers import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
import pickle as pkl
import csv
import copy
import time
import logging
import os
import unittest
import sys
import random
from darknet_ros_msgs.msg import BoundingBoxes
import IPython
import rospy
import cv2
import itertools
from std_msgs.msg import String
from surgeme_models import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs
from yumi_homography_functions import *
from sensor_msgs.msg import CameraInfo
from scipy.spatial import distance
import math
from scipy.interpolate import interp1d
from mrcnn_msgs.msg import ObjMasks
import matplotlib.pyplot as plt


##########################
###     PARAMS         ###
##########################

class Surgeme_Splines():
    def __init__(self, spline_degree=3, coeff_len=6, pole_n=6):
        # Load the data for the left arm
        self.spline_degree = 3
        self.coeff_len = 6
        self.peg_n = 6
        self.model_path = "models"
        self.model = None
        # Load Initial Configuration
        self.y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
        DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
        DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
        self.y.left.set_tool(DELTALEFT)
        self.y.right.set_tool(DELTARIGHT)
        self.y.set_v(40)
        self.y.set_z('z100')
        # Load Neutral Poses
        self.neutral_pose = load_pose_by_path('data/neutral_pose_peg.txt')
        self.neutral_angles = load_pose_by_path('data/neutral_angles_pose_peg1.txt')

    def load_model(self, surgeme_num, arm, model, degree=3):
        with open(self.model_path+'/S'+str(surgeme_num)+'_left_'+model+'_'+str(degree), 'rb') as model_name:
            self.model = pkl.load(model_name)

    def left_close(self):
        self.y.left.close_gripper(force=12,wait_for_res=False)

    def left_open(self):
        self.y.left.move_gripper(0.005)

    def right_close(self):
        self.y.right.close_gripper(force=12,wait_for_res=False)

    def right_open(self):
        self.y.right.move_gripper(0.005)

    def get_curr_pose(self,limb):
        arm = self.y.right if limb == 'right' else self.y.left
        return arm.get_pose()

    def ret_to_neutral(self,limb):
        self.y.set_v(80)
        arm = self.y.right if limb == 'right' else self.y.left
        curr_pos = arm.get_pose()
        target_pose = curr_pos_left
        target_pose.translation = self.neutral_pose[limb].translation
        target_pose.rotation = self.neutral_pose[limb].rotation
        arm.goto_pose(target_pose,False,True,False)
        print "Moved to neutral"

    def ret_to_neutral_angles(self,limb):
        self.y.set_v(80)
        arm = self.y.right if limb == 'right' else self.y.left
        joint_angles = 'left_angles' if limb == 'left' else 'right_angles'
        curr_pos = arm.get_state()
        target_pose = curr_pos
        target_pose.joints = self.neutral_angles[joint_angles].joints
        arm.goto_state(target_pose)
        print "Moved to neutral :)"

    def joint_orient(self,limb,j_val,offset = 150):
        arm = self.y.right if limb == 'right' else self.y.left
        arm_state = arm.get_state()
        if limb == 'left':
            offset = 150
            arm_scale = interp1d([0,180],[-30,offset])
            peg_scale = interp1d([0,offset],[-30,offset])
        else:
            offset = 50
            arm_scale = interp1d([0,180],[-130,offset])
            peg_scale = interp1d([0,offset],[-130,offset])
        joint_angles = arm_state.joints
        gripper_joint = joint_angles[5]
        print ("Value of the gripper joint (joint 5) ",
                gripper_joint)
        if (gripper_joint) - (arm_scale(j_val)) > 0 :
            diff_angle = (gripper_joint)-(arm_scale(j_val))
            joint_angles[5] = gripper_joint-diff_angle
        else:
            diff_angle = (gripper_joint)-(arm_scale(j_val))
            joint_angles[5] = gripper_joint+abs(diff_angle)
        arm_state.joints = joint_angles
        arm.goto_state(arm_state)
        time.sleep(0.5)

    def execute_spline(self, s_init, s_end):
        # go to the init pose
        self.arm.goto_pose(s_init,False,True,False)
        inputs = [s_init.translation]
        inputs.append(s_end.translation)
        inputs = np.array(inputs).reshape(1,-1)
        # predict the path
        pred_waypoints = self.model.predict(inputs)
        # Get the predicted spline
        # add the initial point
        x_way = []
        y_way = []
        z_way = []
        x_way.append(s_init.translation[0])
        y_way.append(s_init.translation[1])
        z_way.append(s_init.translation[2])
        # add the waypoints
        pred_waypoints = pred_waypoints.reshape(self.coeff_len)
        for i in range(self.coeff_len/3):
            x_way.append(pred_waypoints[i*3])
            y_way.append(pred_waypoints[i*3 +1])
            z_way.append(pred_waypoints[i*3 +2])
        # add the endpoint
        x_way.append(s_end.translation[0])
        y_way.append(s_end.translation[1])
        z_way.append(s_end.translation[2])
        # get the spline
        t_points = np.linspace(0,1,20)
        tck_way, u_way = interpolate.splprep([x_way,y_way,z_way ], s=self.spline_degree)
        x_pred, y_pred, z_pred = interpolate.splev(t_points, tck_way)

        fig = plt.figure(2)
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.plot(x_pred, y_pred, z_pred, 'go')
        plt.title('predicted curve')
        plt.show()

        go_to_poses = []
        count = 0
        for x_coord, y_coord, z_coord in zip(x_pred, y_pred, z_pred):
                target_pose = copy.deepcopy(s_init)
                target_pose.translation[0] = x_coord
                target_pose.translation[1] = y_coord
                target_pose.translation[2] = z_coord
                self.arm.goto_pose(target_pose,False,True,False)

    def surgeme1(self,peg,desired_pos,limb):
        self.y.set_v(80)
        self.arm = self.y.right if limb == 'right' else self.y.left
        # Get init and final pose
        init_pose = self.arm.get_pose()
        final_pose = copy.deepcopy(init_pose)
        final_pose.translation = desired_pos
	final_pose.translation[2] += 0.01
        # Load the model
        self.load_model(1, limb, 'regression')
        # Execute the spline
        self.execute_spline(init_pose, final_pose)
        time.sleep(0.2)


    def surgeme2(self,peg,desired_pos,limb):
        self.y.set_v(25)
        self.arm = self.y.right if limb == 'right' else self.y.left
        # Get init and final pose
        init_pose = self.arm.get_pose()
        print "Init location: ", init_pose.translation
        print "Approaching peg number: ",peg
        final_pose = copy.deepcopy(init_pose)
        final_pose.translation = desired_pos
        print("Final Pose for surgeme approach:",final_pose)
        # Load the model
        self.load_model(2, limb, 'regression')
        # Execute the spline
        self.execute_spline(init_pose, final_pose)
        time.sleep(3)
        # Close the gripper
        if limb == 'left':
                self.left_close()
        else:
                self.right_close()

    def surgeme3(self,peg,limb):#lift is hardcoded
	self.y.set_v(80)
	arm = self.y.right if limb == 'right' else self.y.left
        # Get init and final pose
        init_pose = self.arm.get_pose()
        print "Init location: ", init_pose.translation
        final_pose = copy.deepcopy(init_pose)
	final_pose.translation[2] += 0.03
        print("Final Pose for surgeme approach:",final_pose)
        # Load the model
        self.load_model(3, limb, 'regression')
        # Execute the spline
        self.execute_spline(init_pose, final_pose)
	#print "Desired_POS",desired_pos
	time.sleep(1)

