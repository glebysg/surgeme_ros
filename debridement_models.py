import logging
import time
import os
import unittest
import numpy as np
import copy
import sys
import random
import pickle as pkl
import csv
import IPython
from helpers import load_pose_by_path
from scipy.interpolate import interp1d
import math
import json 
import socket
import threading
from scipy.spatial.transform import Rotation as R

MEGA = 1000000.0
##########################################################################################################################
class Debridement_Models():
    def __init__(self):
        self.a = 2
        # init socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.local_address = ('128.46.103.192', 9753)
        self.sock.bind(self.local_address)
        self.robot_address = ("128.46.103.195", 8642)
        self.left_data = None
        self.right_data = None


    def get_pose(self):
        # while True:
        data, server = self.sock.recvfrom(4096)
        if data:    
            pkg = json.loads(data)
            current_state=pkg.get("current_state", -1)
            left_data=pkg.get("arm0", "").split(" ")
            right_data=pkg.get("arm1", "").split(" ")
            if len(left_data)==44:
                self.left_data = np.array(left_data,dtype=np.float)
                # print("left data", left_data)
            if len(right_data)==44:
                self.right_data = np.array(right_data,dtype=np.float)
                # print("right data", self.right_data[8:20])
            return 1
        else:
            return 0
    
    def slepo(self, t, dt=0.01):
        cnt = 0
        while cnt<(t/dt):
            cnt+=1
            time.sleep(dt)

    def move_to_pose(self,pos, rotation=None,limb='right'):
        global MEGA
        if limb == 'right':
            
            pkg = {}        
            print("pitch", self.right_data[27]/MEGA,"yaw", self.right_data[28]/MEGA, "roll", self.right_data[29]/MEGA)
            pkg["pos0"] = [int(p) for p in self.left_data[24:27]] # Current position x,y,z of the robot
            print("Position ", self.right_data[24:27])
            print("Position other ", self.right_data[11],self.right_data[15], self.right_data[19])

            print("Position from Mask ", [int(p*1000000) for p in pos])

            pkg["pos1"] = [int(p*1000000) for p in pos] #[int(p - (ind%2)*16000) for ind, p in  enumerate(self.right_data[24:27])] # change position from meters to microns
            pkg["gripper0"] = 0 #int(self.left_data[30]) # current opening
            pkg["gripper1"] = 50 #int(self.right_data[30]) #Gripper open 
            # rot_matrix = R.from_euler("ZYX", rot, degrees=True).as_dcm().flatten()
            pkg["rot0"] = [self.left_data[i]/MEGA for i in [8,9,10,12,13,14,16,17,18]] # current rotation
            if rotation is None:
                pkg["rot1"] = [self.right_data[i]/MEGA for i in [8,9,10,12,13,14,16,17,18]]# list(rot_matrix)
            else:
                pkg["rot1"] = [r/MEGA for r in rotation]# list(rot_matrix)

            pkg["movement_time"] = 0.5
            pkt = json.dumps(pkg)
            pkt = str.encode(pkt)
            print(pkt)
            self.sock.sendto(pkt, self.robot_address)
        else:
            pkg = {}        
            pkg["pos1"] = [int(p) for p in self.right_data[24:27]] # Current position x,y,z of the robot
            pkg["pos0"] = [int(p*1000000) for p in pos] # change position from meters to microns
            pkg["gripper1"] = int(self.right_data[30]) # current opening
            pkg["gripper0"] = int(90) # Gripper open
            # rot_matrix = R.from_euler("ZYX", rot, degrees=True).as_dcm().flatten()
            pkg["rot1"] = [self.right_data[i] for i in [8,9,10,12,13,14,16,17,18]] # current rotation
            pkg["rot0"] = [self.left_data[i] for i in [8,9,10,12,13,14,16,17,18]]
            pkg["movement_time"] = 0.5
            pkt = json.dumps(pkg)
            pkt = str.encode(pkt)
            print(pkt)
            self.sock.sendto(pkt, self.robot_address)
        self.slepo(5)
    

    def approach(self):
        pkg = {}
        pkg["cmd"] = "Approach"
        pkt = json.dumps(pkg)
        pkt = str.encode(pkt)
        print(pkt)
        self.sock.sendto(pkt, self.robot_address)
    
    def stow(self):
        pkg = {}
        pkg["cmd"] = "Stowed"
        pkt = json.dumps(pkg)
        pkt = str.encode(pkt)
        print(pkt)
        self.sock.sendto(pkt, self.robot_address)

    def ready(self):
        pkg = {}
        pkg["cmd"] = "Safe"
        pkt = json.dumps(pkg)
        pkt = str.encode(pkt)
        print(pkt)
        self.sock.sendto(pkt, self.robot_address)


    def S1(self, pose, rot):
        self.move_to_pose(pose, None, limb="right")
        self.slepo(2)        

    def S2(self, pose, limb="right"):
        # TODO REMOVE BECAUSE ITS HARDCODED
        target_rot = np.array([-716859,   147192,   681504,   403274,
                                -305305,  -945036,  -117033, 28733.9,  
                                626819,  -291963,   722396,   201184 ], dtype=np.float).reshape(-1,4)
        target_pose = [pose[0], pose[1], target_rot[-1,-1]]
        target_euler_rot = np.array([-677464,-2738966,-384086], dtype=float)
        self.move_to_pose(target_pose, target_rot[:3,:].flatten(), limb="right")


    def stop(self):
        self.ready()
        self.slepo(10)
        self.approach()
        self.slepo(10)
        self.stow()
        self.slepo(10)