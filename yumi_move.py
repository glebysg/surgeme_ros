
import os, time, sys
import rospy
from sensor_msgs.msg import Image
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
import cv2
import numpy as np
import quaternion
from cv_bridge import CvBridge, CvBridgeError
from autolab_core import RigidTransform
from helpers import load_pose_by_path
from collections import deque
from std_msgs.msg import Float32MultiArray

sys.path.append('/home/isat/Forward/forward-simulation')
from hydra_capture import HydraIn
import argparse


class YumiMove():
    def __init__(self):
        self.neutral_angles = load_pose_by_path('data/neutral_joints_straight.txt')

        self.y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)

        self.ret_to_neutral_angles('left')
        self.ret_to_neutral_angles('right')
        self.gripper_open(limb="left")
        self.gripper_open(limb="right")
        time.sleep(0.5)

        self.l_diff = np.array([0,0,0,1,0,0,0],dtype=np.float64)
        self.r_diff = np.array([0,0,0,1,0,0,0],dtype=np.float64)
        self.gripper_poses = Float32MultiArray(data=[0.05,0.05])

        rospy.Subscriber('/ldiff',Float32MultiArray, self.ldiff_sub)
        rospy.Subscriber('/rdiff',Float32MultiArray, self.rdiff_sub)
        rospy.Subscriber('/grip',Float32MultiArray, self.grip_sub)

            # r_diff = rospy.wait_for_message('/rdiff',Float32MultiArray) 
            # gripper_poses = rospy.wait_for_message('/grip',Float32MultiArray))

    def ret_to_neutral_angles(self,limb):
        self.y.set_v(80)
        limb_angles = 'left_angles' if limb == 'left' else 'right_angles'
        self.goto_joint_state(self.neutral_angles[limb_angles].joints,limb)
        time.sleep(1.0)

    def goto_joint_state(self,joints,limb):
        arm = self.y.right if limb == 'right' else self.y.left
        joint_state = YuMiState(vals=joints)
        arm.goto_state(joint_state,False)       

    def get_curr_pose(self,limb):
        arm = self.y.right if limb == 'right' else self.y.left
        return arm.get_pose()

    def gripper_close(self, limb):
        arm = self.y.left if limb == 'left' else self.y.right
        arm.close_gripper(force=12)
        time.sleep(0.01)

    def gripper_open(self, gripper_value=0.005, limb='right'):
        arm = self.y.left if limb == 'left' else self.y.right
        arm.move_gripper(gripper_value)
        time.sleep(0.01)


    def ldiff_sub(self, l_off):
        self.l_diff[:3] += l_off.data[:3]
        l_rot = np.quaternion(*l_off.data[3:])*np.quaternion(*self.l_diff[3:])
        self.l_diff[3:] = quaternion.as_float_array(l_rot)

    def rdiff_sub(self, r_off):
        self.r_diff[:3] += r_off.data[:3]
        r_rot = np.quaternion(*r_off.data[3:])*np.quaternion(*self.r_diff[3:])
        self.r_diff[3:] = quaternion.as_float_array(r_rot)

    def grip_sub(self, grip):
        self.gripper_poses = grip

    def loop(self):
        self.y.set_v(800)
        self.y.set_z("fine")
        rospy.init_node('move_yumi',anonymous=True)
        r_closed = False
        l_closed = False
        
        while not rospy.is_shutdown():
            # print('Hello')
            stim = time.time()
            
            l_diff_poses = np.copy(self.l_diff)
            l_rot = np.quaternion(*l_diff_poses[3:])
            r_diff_poses = np.copy(self.r_diff)
            r_rot = np.quaternion(*r_diff_poses[3:])
            # print(gripper_poses.data)
            # print(l_diff_poses)
            
            l_gripper_pose = self.gripper_poses.data[0]
            r_gripper_pose = self.gripper_poses.data[1]


            self.l_diff = np.array([0,0,0,1,0,0,0],dtype=np.float64)
            self.r_diff = np.array([0,0,0,1,0,0,0],dtype=np.float64)

            
            # Process left pose
            current_pose_l = self.y.left.get_pose()
            
            current_pose_l.translation += l_diff_poses[:3]*0.50
            target_rotation_l = np.quaternion(*current_pose_l.quaternion)*(l_rot)
            target_rotation_inter = quaternion.slerp_evaluate(np.quaternion(*current_pose_l.quaternion),target_rotation_l,0.5)
            

            # Convert rotation to matrix
            traget_rotation_mat = quaternion.as_rotation_matrix(target_rotation_inter)
            new_pose_l = RigidTransform(rotation=traget_rotation_mat,
                                 translation=current_pose_l.translation)
            
            #--------------------------
            # Process right pose 
            current_pose_r = self.y.right.get_pose()
            # CHANGE
            # current_pose_r = current_pose_l
            
            current_pose_r.translation += r_diff_poses[:3]*0.50
            target_rotation_r = np.quaternion(*current_pose_r.quaternion)*(r_rot)
            target_rotation_inter = quaternion.slerp_evaluate(np.quaternion(*current_pose_r.quaternion),target_rotation_r,0.5)
            
            # Convert rotation to matrix
            traget_rotation_mat = quaternion.as_rotation_matrix(target_rotation_inter)
            new_pose_r = RigidTransform(rotation=traget_rotation_mat,
                                 translation=current_pose_r.translation)
            # print(l_gripper_pose, l_closed)
            # print(r_gripper_pose, r_closed)
            # Set poses

            if r_gripper_pose == 0 and r_closed==False:
                r_closed=True
                # print('closing..') 
                self.gripper_close('right')

            elif r_gripper_pose>0 and r_closed:
                self.gripper_open(r_gripper_pose,"right")
                print('here - ropen')
                r_closed=False

            if l_gripper_pose == 0 and l_closed==False:
                l_closed=True
                # print('closing..') 
                self.gripper_close('left')

            elif l_gripper_pose>0 and l_closed:
                self.gripper_open(l_gripper_pose,"left")
                print('here - lopen')
                l_closed=False

            # print(new_pose_l)

            # CHANGE
            self.y.left.goto_pose(new_pose_l,False,True,False)
            self.y.right.goto_pose(new_pose_r,False,True,False)
            # print(time.time()-stim)        

if __name__ == '__main__':
    ym = YumiMove()
    ym.loop()