import logging
import time
import os
import unittest
import numpy as np
import copy
import sys
from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
import IPython
import argparse
import pickle as pkl

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action="store", dest="arm", default="left",
            help="record arm. Possible options: left, right or both")
    parser.add_argument('-s', action="store", dest="filename", default="pose",
            help="name of the pkl file where the object is recorded")
    parser.add_argument('--joints',action="store_true", default=False,
            help="save joint values instead of gripper pose")

    args = parser.parse_args()
    save_joints = False if not args.joints else True

    # Initialize Yumi
    y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)

    #setup the ool distance for the surgical grippers
    # ORIGINAL GRIPPER TRANSFORM IS tcp2=RigidTransform(translation=[0, 0, 0.156], rotation=[[ 1. 0. 0.] [ 0. 1. 0.] [ 0. 0. 1.]])
    DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
    DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
    y.left.set_tool(DELTALEFT)
    y.right.set_tool(DELTARIGHT)
    y.set_v(40)
    y.set_z('z100')


    # Get Poses
    if save_joints:
        poses = {'left_angles': None, 'right_angles':None}
        if args.arm == "left" or args.arm == "both":
            poses['left_angles']=y.left.get_state()
        if args.arm == "right" or args.arm == "both":
            poses['right_angles']=y.right.get_state()
    else:
        poses = {'left': None, 'right':None}
        if args.arm == "left" or args.arm == "both":
            poses['left']=y.left.get_pose()
        if args.arm == "right" or args.arm == "both":
            poses['right']=y.right.get_pose()

    # save pkl object
    with open(args.filename, "wb") as output_file:
        pkl.dump(poses,output_file)

if __name__ == '__main__':
    main()

