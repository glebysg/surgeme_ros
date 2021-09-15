import logging
import time
import os
import unittest
import numpy as np
import copy
import sys
import random
import pickle as pkl
import rospy
import csv
import IPython
import cv2
import itertools
from helpers import *
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
from debridement_models import *
from forwardcomm.forward_comm_robot_terminal import robotTerminal
from collections import OrderedDict
import argparse
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from mrcnn_msgs.msg import ObjMasks
from std_msgs.msg import Empty

#envparse = argparse.ArgumentParser()
#envparse.add_argument("-s", "--subject_number", action='store', type=int, help="subject number", required=True)
#envparse.add_argument("-s", "--trial_counter", action='store', type=int, help="trial counter", required=True)
#args = envparse.parse_args()

#trial_ctr = args.trial_counter
#sub_number = args.subject_number

class Scene():
    #######################
    #        INIT         #
    #######################
    def __init__(self,exec_model):
        # manual params
        self.num_debri = 3
        self.bridge = CvBridge()
        self.K = np.identity(3)
        self.depth_vals = []
        self.color_frame =[]
        self.mask_frames =[]
        self.obj_bbs = []
        self.pose_succ = False
    

        # offsets
        self.grasp_height = 0.17#Constant depth to the surface (blood: 0.022, regular: 0.0179)
        self.left_grasp_offset = 2
        self.right_grasp_offset = 2
        self.approach_height = 0.05
        self.drop_offset = [0.002,0.004,-0.015]
        self.drop_height = 0.045
        self.ksize = (3,3)

        self.exec_model = exec_model
        
        # Flags
        self.mask_flag = False

    ########################
    # Subscriber           #
    ########################

    # All ROS the subscribers
    def subscribe(self):
        rospy.init_node('Debridement_execution', anonymous=True)
        # rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.get_bbox) #This is tthe subscriber for the darknet bounding boxes. SInce we use mask we dont need this
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb)
        rospy.Subscriber("/camera/color/camera_info",CameraInfo, self.camera_callback)
        rospy.Subscriber("/camera/color/image_raw",Image, self.image_callback)
        rospy.Subscriber("/masks_t",ObjMasks,self.mask_callback)
        # rospy.Subscriber("/darknet_ros/tracked_bbs", TrackedBoundingBoxes,self.pole_cb)
        rospy.Subscriber('/posepub', Empty, self.pose_cb)

    ########################
    # Callback functions #
    ########################
    # def get_bbox(self, data):
    #     count = 0
    #     bounds = data.bounding_boxes
    #     for box in data.bounding_boxes:
    #         if box.Class == "peg":
    #             pegs.append([box.xmin, box.xmax, box.ymin, box.ymax])
    #     self.pegs = pegs

    def depth_cb(self,data):
        try:
            data.encoding = "mono16"
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono16")
        except CvBridgeError as e:
            print(e)

        (rows,cols) = cv_image.shape
        self.depth_vals = cv_image/1000.0

        # vision_logger.info('Recevied Depth Image from ROS')


    def camera_callback(self, data):
        self.K = np.array(list(data.K)).reshape(3,3)

    def image_callback(self, data):
        self.color_frame = self.bridge.imgmsg_to_cv2(data, "rgb8")
        # self.color_frame = cv2.cvtColor(color_frame,cv2.COLOR_BGR2RGB)
        # vision_logger.info('Received RGB Image from ROS')

    # UPDATE object masks
    def mask_callback(self,data):
        pegs = []
        masks = []
        for box, mask in zip(data.bbs, data.masks):
                pegs.append([box.data[0], box.data[2], box.data[1], box.data[3]])
                mask_frame = self.bridge.imgmsg_to_cv2(mask,'passthrough')
                mask_frame = cv2.cvtColor(mask_frame,cv2.COLOR_BGR2RGB)
                masks.append(mask_frame)
        # if len(pegs)==len(self.trk_pegs) and len(pegs)==self.num_triangles:
        #     ids = self.reorder_bbs(pegs, self.trk_pegs)
        #     self.pegs = [pegs[i] for i in ids] #here saved ROIS of triangles
        #     self.mask_frames = [masks[i] for i in ids]
        # else:
        self.mask_flag = True
        self.obj_bbs = pegs #here saved ROIS of triangles
        self.mask_frames = masks


    def pose_cb(self,data):
        self.pose_succ = self.exec_model.get_pose()
        
            # vision_logger.info('Published left and right Gripper data')
    ################################################################
    #                               Utils                          #
    ################################################################
    def l2dis(self, pt1, pt2):
        return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

    def l2vec(self, cpt, pts):
        l2vec = []
        for pt in pts:
            l2vec.append(self.l2dis(cpt,pt))
        return l2vec

    def get_mindis_point(self, cpt, pts):
        dists = []
        for i in range(pts.shape[0]):
            dists.append(np.linalg.norm(np.array(cpt) - pts[i,:]))

        min_id = np.argsort(dists)[0]
        # print(dists)
        return(pts[min_id,:])

    def get_maxdis_point(self, cpt, pts):
        dists = []
        for i in range(pts.shape[0]):
            # print(pts[i,0,:])
            # print(np.array(cpt))
            dists.append(np.linalg.norm(np.array(cpt) - pts[i,0,:]))
            # print('-'*10)
        max_id = np.argsort(dists)[-1]
        return(pts[max_id,0,:])

    def rotatept(self, pt, cpt, ang):
        rot_pt = [0,0]
        rot_pt[0] = int((pt[0]-cpt[0])*np.cos(ang) + (pt[1]-cpt[1])*np.sin(ang) + cpt[0])
        rot_pt[1] = int(-(pt[0]-cpt[0])*np.sin(ang) + (pt[1]-cpt[1])*np.cos(ang) + cpt[1])
        return rot_pt

    def dis2pts(self, cpt, pts):
        dists = []
        for i in range(pts.shape[0]):
            # print(pts[i,0,:])
            # print(np.array(cpt))
            dists.append(np.linalg.norm(np.array(cpt) - pts[i,0,:]))
            # print('-'*10)
        min_id = np.argsort(dists)[0]
        return(pts[min_id,0,:])

    #######################################################
    #                State computation                    #
    #######################################################

    def grip_angle(self, centroid):
        # dX = (pole[1]-corner[1])
        # dY = -(pole[0]-corner[0])
        # print("Dx, Dy", dX, dY)
        #
        # angle = math.atan2(dY, dX)
        # angle = math.degrees(angle)
        # if angle<0:
        #     angle = angle + 180
        # print('Orientation tan2', angle)
        angle = 0
        return angle


    def grasp_points(self, K, offset=1):
        if limb=="left":
            offset=self.left_grasp_offset
        elif limb=="right":
            offset=self.right_grasp_offset

        depth_ROI = self.depth_vals#[bbox[0]:bbox[1],bbox[2]:bbox[3]]

        img = self.mask_frames[0].copy()       # Change on detect 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        bw_img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]

        # cv2.imshow('Bin', bw_img)
        # cv2.waitKey(1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.ksize)
        bw_img = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow('Bin_Morphed', bw_img)
        # cv2.waitKey(1)

        h,w = gray.shape
        h = h/2
        w = w/2

        #cimg,cons,h
        cons,_ = cv2.findContours(bw_img , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i,c in enumerate(cons):
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # cv2.circle(ROI, (cX, cY), 3, (0, 0, 255), -1)
            #cv2.putText(ROI, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # ROI = cv2.drawContours(ROI, cons, -1, (255,0,0), 1)
        # cv2.imshow('Alternate points', ROI)
        # cv2.waitKey(0)


        grasp_point = np.array([cX, cY])
        corner_point = grasp_point.copy()#np.array([0,0])
        angle = self.grip_angle(grasp_point)

        # Translate points  to full image frame
        # grasp_point[0] = grasp_point[0]#+bbox[2]
        # grasp_point[1] = grasp_point[1]#+bbox[0]
        # corner_point[0] = bbox[2]
        # corner_point[1] = bbox[0]
        # print('Trinagle corner to go to ', corner_point)

        # Convert corner and grasp point to robot space
        corner_point = np.array(corner_point)
        z_coord = get_max_depth(depth_ROI)
        yumi_corner_pose = cam2robot(corner_point[0], corner_point[1], z_coord , K,limb,robot='Taurus')
        yumi_corner_pose[2] = self.approach_height

        grasp_point = np.array(grasp_point)
        print('Grasp point' , grasp_point)
        z_coord = get_min_depth(depth_ROI)
        yumi_grasp_pose = cam2robot(grasp_point[0], grasp_point[1],z_coord,K,limb,robot='Taurus')
        yumi_grasp_pose[2] = self.grasp_height
        print(yumi_grasp_pose)

        grasp_centroid = cam2robot(cX, cY,z_coord,K,limb,robot='Taurus')
        return yumi_grasp_pose,angle,yumi_corner_pose, grasp_centroid

######################################################################
############################# EXECUTION ##############################
######################################################################
def exit_routine():
    global execution
    # cv2.destroyAllWindows()
    execution.stop()
    print("exiting")
    # exit(0)

#log_fh = logging.FileHandler('log_folder/robot_execution_terminal_subject_{0}_trial_{1}.log'.format(sub_number, trial_ctr ))
# log_fh = logging.FileHandler('log_folder/robot_execution_terminal_{0}.log'.format(str(int(time.time()))))
# log_fh.setLevel(logging.DEBUG)
# log_format = logging.Formatter('%(created)f - %(asctime)s - %(name)s - %(message)s')
# log_fh.setFormatter(log_format)

# surgeme_logger = logging.getLogger('robot_terminal.execution_module')
# surgeme_logger.setLevel(logging.DEBUG)
# surgeme_logger.addHandler(log_fh)

# log_fh = logging.FileHandler('log_folder/robot_vision_terminal_{0}.log'.format(str(int(time.time()))))
# log_fh.setLevel(logging.DEBUG)
# log_format = logging.Formatter('%(created)f - %(asctime)s - %(name)s - %(message)s')
# log_fh.setFormatter(log_format)

#vision_logger = logging.getLogger('robot_terminal.vision_module')
#vision_logger.setLevel(logging.DEBUG)
#vision_logger.addHandler(log_fh)

if __name__ == '__main__':
    # Start Robot Environment
    # Start surgemes class and open grippers and take robot to neutral positions
    global robTerminal
    global execution
    robot="taurus"
    execution = Debridement_Models()
    robTerminal = robotTerminal(debug = False)
    execution.approach()
    execution.slepo(5)
    execution.ready()
    execution.slepo(5)

    ROI_offset = 10 #ROI bounding box offsets
    # Start the Scene calss to obtain pole positions and bounding boxes etc.
    scene = Scene(execution)
    scene.subscribe()
    stop = 0
    surgeme_no = 0
    count = 0
    # Open the robot arms
    #### TODO select limb from messages
    limb = 'right'
    opposite_limb = 'right' if limb == 'left' else 'left'

    # EXECUTION GET TO HOME POSE
    #execution.arm_open('left')
    #time.sleep(0.5)
    #execution.arm_open('right')
    #time.sleep(0.5)
    #execution.ret_to_neutral_angles(limb)
    #time.sleep(0.5)
    #execution.ret_to_neutral_angles(opposite_limb)

    rospy.on_shutdown(exit_routine)
    ### Initial setup complete
    label_thesholds = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
    prev_label = -1
    prev_obj_num = -1
    equivalent_labels = [[1],[2],[3],[4],[5],[6],[7]]
    prev_execution_limb = 'none'
    grasp_location = None

    # Wait for initial detecions before start
    print("Waiting for mask...")
    while not scene.mask_flag:
        continue
    print("Received first mask, continuing..")

    while not rospy.is_shutdown():
        ################ MESSAGE RETRIEVING
        # message = robTerminal.getSurgemeMsg()
        # if message == "Surgeme Queue Empty" or len(message)!=2:
        #     # print('Flag')
        #     continue
        # # if there is too much delay this mesage is old
        # ##### Parse the message:
        # delay = int(message[1])
        # ###########################################################################################################
        # #                                                                                                         #
        # #  |frame id| current surgeme label| predection probability vector| object number| limb | pedal pressed|  #
        # #                                                                                                         #
        # ###########################################################################################################
        # frame_id = int(message[0].split(":")[0])
        # label = int(message[0].split(":")[1])+1
        # pred_prob = []
        # for i in range(7):
        #     pred_prob.append(float(message[0].split(":")[2].split(" ")[i]))

        # obj_num = int(message[0].split(":")[3])
        # limb = message[0].split(":")[4]
        # pedal_pressed = int(message[0].split(":")[5])

        # pred_prob = pred_prob[label-1]

        # surgeme_logger.info('Received with delay:'+str(message[1])+':'+message[0])
        # #surgeme_logger.info('Frame Id {0}, Surgeme {1}, probability {2} received from Simulator with Delay {3}'.format(frame_id, label, pred_prob, message[1]))
        # #surgeme_logger.info('Surgeme params: obj_num: {0}, limb: {1}, pedal_pressed: {2}'.format(obj_num, limb, pedal_pressed))

        # if delay>5:
        #     print("Following surgeme ignored:")
        #     print("Message: \"{0}\" Delay {1}".format(message[0], message[1]))
        #     surgeme_logger.info('Ignorinng at, Frame Id {0}, Surgeme {1}, Probability {2}, Delay {3}, High Delay'.format(frame_id, label, pred_prob, message[1]))
        #     continue
        # # else:
        #     # print("surgeme", label, "probability:", pred_prob, "On object: ", obj_num)
        # # check the execution threshhold
        # if pred_prob < label_thesholds[label-1] or (prev_label == label and prev_obj_num == obj_num) \
        #   or not pedal_pressed or limb == 'none':
        #     print("ignoring with the first condition",label,pred_prob,limb, pedal_pressed)
        #     surgeme_logger.info('Ignorinng at, Frame Id {0}, Surgeme {1}, Probability {2}, Delay {3}, object {4}, Condition: 1'.format(frame_id, label, pred_prob, message[1], obj_num))
        #     continue
        # # if the prediction is above the threshold and different from the previous one:
        # # check that they are not equivalent
        # if prev_label in equivalent_labels[label-1]:
        #     print("ignoring with the second condition",prev_label, label,pred_prob,limb)
        #     surgeme_logger.info('Ignorinng at, Frame Id {0}, Surgeme {1}, Probability {2}, Delay {3}, object {4}, Condition: 2'.format(frame_id, label, pred_prob, message[1], obj_num))
        #     continue
        # # else:
        #     # print("executing", label,pred_prob)
        #     # continue

        # print("surgeme", label, "probability:", pred_prob, "On object: ", obj_num, "with arm", limb)

        # #### TODO select limb from messages
        # opposite_limb = 'right' if limb == 'left' else 'left'
        # execution.execution.ROBOT_BUSY = True
        # rob_pose = execution.get_curr_pose(limb)
        # time.sleep(0.1)
        # execution.execution.ROBOT_BUSY = False

        # selected_pole = obj_num
        # drop_pole_num = None

        # #### TODO decide of to read message or if to execute surgeme :)
        # # time.sleep(3)
        # # surgeme_no = input('Enter the surgeme number youd like to perform: ')
        # surgeme_no = label
        # surgeme_logger.info('Executing at, Frame Id {0}, Surgeme {1}, Probability {2}, Delay {3}, Obj id {4}'.format(frame_id, label, pred_prob,message[1], obj_num))

        ####################### MESSAGE RETRIEVING ######################################3
        centroid = None
        for surgeme_no in range(1,3):
            if surgeme_no == 1:
                # Calculate target
                target_location, _, _, centroid = scene.grasp_points(scene.K)# obtain corner points and grasp poiints from scene 
                print(target_location)
                execution.ready()
                execution.slepo(2)
                execution.S1(target_location,limb)#Perform Approach
                print("Performed approach")
                # surgeme_logger.info('Executed Surgeme 1 on object ID: {0}'.format(obj_num))
            if surgeme_no == 2:
                if centroid is not None:
                    target_location, _, _, centroid = scene.grasp_points(scene.K)# obtain centroid from scene
                execution.S2(centroid,limb)#Perform Grasp
                centroid = None
                execution.slepo(2)
                rospy.signal_shutdown('Hello')
                # surgeme_logger.info('Executed Surgeme 2')
            if surgeme_no == 3:
                if grasp_location is not None:
                    execution.S3(limb)#Perform Lift
                    # surgeme_logger.info('Executed Surgeme 3')
            if surgeme_no == 4:
                # if the limb changes just for the transfer, keep
                # the one use previously since it's more likely to
                # be correct
                # if prev_execution_limb != limb:
                #     limb = prev_execution_limb
                execution.S4(limb)#Perform Go To transfer
                # surgeme_logger.info('Executed Surgeme 4')

            if surgeme_no == 5:
                transfer_flag = execution.S5(limb, opposite_limb)
                # surgeme_logger.info('Executed Surgeme 5')
            if surgeme_no == 6:
                execution.S6(drop_pt,limb)#Perform Approach
                # surgeme_logger.info('Executed Surgeme 6 on object ID: {0}'.format(obj_num))
            if surgeme_no == 7:
                execution.S7(limb,opposite_limb,scene.drop_height)#Perform Drop
                # surgeme_logger.info('Executed Surgeme 7')
            count = count+1
            # prev_label = label
            # prev_obj_num = obj_num
            # prev_execution_limb = limb

        # exit(0)
    # rospy.spin()
