import logging
import time
import os
import unittest
import numpy as np
import copy
import sys
import random
from darknet_ros_msgs.msg import BoundingBoxes, TrackedBoundingBoxes
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
import csv
from surgeme_wrapper import Surgemes
from geometry_msgs.msg import Pose
import datetime

class Scene():
    #######################
    #        INIT         #
    #######################
    def __init__(self,exec_model):
        self.grasp_height = 0.0195#Constant depth to the surface (blood: 0.022, regular: 0.0179)
        self.left_grasp_offset = 2
        self.right_grasp_offset = 2
        self.approach_height = 0.04 
        self.drop_offset = [0.003,-0.005,0] 
        # grasp point params
        self.alpha = 0.70
        self.ksize = (3,3)
        ##### manual params ########
        self.pegs = []
        self.poles_found = []
        self.bridge = CvBridge()
        self.KI = []
        self.depth_vals = []
        self.color_frame =[]
        self.mask_frames =[]
        self.pole_flag = 0
        self.exec_model = exec_model

    ########################
    # Subscriber callbacks #
    ########################
    def get_bbox(self,data):
        count = 0
        bounds = data.bounding_boxes
        for box in data.bounding_boxes:
            if box.Class == "peg":
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

    def camera_callback(self, data):
        self.K = np.array(list(data.K)).reshape(3,3)

    def image_callback(self, data):
        color_frame = self.bridge.imgmsg_to_cv2(data, "rgb8")
        self.color_frame = cv2.cvtColor(color_frame,cv2.COLOR_BGR2RGB)
        # cv2.imshow('image',self.color_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def mask_callback(self,data):
        pegs = []
        masks = []
        for box, mask in zip(data.bbs,data.masks):
                pegs.append([box.data[0], box.data[2], box.data[1], box.data[3]])
                mask_frame = self.bridge.imgmsg_to_cv2(mask,'passthrough')
                mask_frame = cv2.cvtColor(mask_frame,cv2.COLOR_BGR2RGB)
                masks.append(mask_frame)
        self.pegs = pegs #here saved ROIS of triangles
        self.mask_frames = masks
        # mask_img = data.masks
        # print(mask_img.encoding)
        # if len(data.masks)>0:
        #   if len(data.masks[0].data)>0:
        #       mask_frame = self.bridge.imgmsg_to_cv2(data.masks[0],'passthrough')
        #       self.mask_frame = cv2.cvtColor(mask_frame,cv2.COLOR_BGR2RGB)
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

########################### End of all callback functions #########################
    # Retuns the orientation angle for grasping in the imgage
    # frame.This funcion is called in peg_grasp_points
    def grip_angle(self,grip_corners):
        print ("Grip Corners:",grip_corners)
        corner1 = grip_corners[0]
        corner2 = grip_corners[1]
        dY = (corner1[1]-corner2[1])
        dX = (corner1[0]-corner2[0])
        print("Dx, Dy", dX, dY)
        if dX == 0:
            angle = 90
        else:
            angle = math.atan(float(dY)/float(dX))
            angle = math.degrees(angle)
        if dX<0 and dY<0 :
            angle = angle
        elif dX<0 or dY<0:
            angle = angle+180

        print ("Orientation Angle:",(angle))
        return angle

    def alt_grip_angle(self, pole, corner):
        dX = (pole[1]-corner[1])
        dY = -(pole[0]-corner[0])
        print("Dx, Dy", dX, dY)
        
        angle = math.atan2(dY, dX)
        angle = math.degrees(angle)
        if angle<0:
            angle = angle + 180
        print('Orientation tan2', angle)

        # if dX == 0:
        #     angle = 0
        # else:
        #     angle = math.atan(float(dY)/dX)
        #     angle = math.degrees(angle)
        
        # if dX<0 and dY<0 :
        #     angle = angle
        # elif dX<0 or dY<0:
        #     angle = angle+180

        # print ("Orientation Angle:",(angle))
        return angle

    
    def rotatept(self, pt, cpt, ang):
        rot_pt = [0,0]
        rot_pt[0] = int((pt[0]-cpt[0])*np.cos(ang) + (pt[1]-cpt[1])*np.sin(ang) + cpt[0])
        rot_pt[1] = int(-(pt[0]-cpt[0])*np.sin(ang) + (pt[1]-cpt[1])*np.cos(ang) + cpt[1])
        return rot_pt

    def l2dis(self, cpt, pts):
        dists = []
        for i in range(pts.shape[0]):
            # print(pts[i,0,:])
            # print(np.array(cpt))
            dists.append(np.linalg.norm(np.array(cpt) - pts[i,0,:]))
            # print('-'*10)
        min_id = np.argsort(dists)[0]
        return(pts[min_id,0,:])

    def get_mindis_point(self, cpt, pts):
        dists = []
        for i in range(pts.shape[0]):
            dists.append(np.linalg.norm(np.array(cpt) - pts[i,:]))
        
        min_id = np.argsort(dists)[0]
        print(dists)
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

    def get_edge_pts(self, min_pt, cpt, pts):
        # Get first rotated pt
        pt1 = self.rotatept(min_pt,cpt, 2*np.pi/3)
        # Find closest point on contour
        min_pt1 = self.l2dis(pt1, pts)

        # Get second point
        pt2 = self.rotatept(min_pt,cpt, -2*np.pi/3)
        min_pt2 = self.l2dis(pt2, pts)

        return np.array([min_pt, min_pt1, min_pt2])

    def alternate_grasp_points(self, ROI, bbox, limb, K, offset=1):
        if limb=="left":
            offset=self.left_grasp_offset
        elif limb=="right":
            offset=self.right_grasp_offset

        depth_ROI = self.depth_vals[bbox[0]:bbox[1],bbox[2]:bbox[3]]

        img = ROI.copy()
        gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
        bw_img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]
        
        cv2.imshow('Bin', bw_img)
        cv2.waitKey(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.ksize)
        bw_img = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('Bin_Morphed', bw_img)
        cv2.waitKey(0)

        h,w = gray.shape
        h = h/2
        w = w/2

        #cimg,cons,h 
        cons,_ = cv2.findContours(bw_img , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i,c in enumerate(cons):
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(ROI, (cX, cY), 3, (0, 0, 255), -1)
            #cv2.putText(ROI, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Get the three edge points
            pt1 = self.l2dis([cX, cY], c)
            corners = np.array(self.get_edge_pts(pt1, [cX, cY], c))

            # Plot the points
            for pt in corners:
                cv2.line(ROI, (cX,cY), (pt[0],pt[1]), (0,0,255), 1)
                cv2.circle(ROI, (pt[0],pt[1]), 2, (0, 0, 255), -1)
        ROI = cv2.drawContours(ROI, cons, -1, (255,0,0), 1)
        cv2.imshow('Alternate points', ROI)
        cv2.waitKey(0)
        print('*'*50)
        print('Cornersss', corners)

        # Get traingle corners
        pt_corner1 = self.get_maxdis_point([cX,cY], c)
        tri_corners = np.array(self.get_edge_pts(pt_corner1, [cX, cY], c))
        for  tri_pt in tri_corners:
            cv2.circle(img,(tri_pt[0],tri_pt[1]),2,(0,255,255),-1)


        pole = np.array([cX, cY])
        gpoints = []
        corner_pairs = np.array([[[0,0],[0,0]]])
        alpha = self.alpha
        # Get the actual grasping points, inside the object of interest
        for cor in corners:
            gpoints.append(pole*(1-alpha) + cor*alpha)

        gpoints = np.array(gpoints, dtype=np.int)
        print("Grasp Points",gpoints)

        for coords in gpoints:
            cv2.circle(img,(coords[0],coords[1]),1,(255,0,0),-1)
        
        if limb == 'left':
            grasp_point,grasp_index,corner_point,corner_index = min_dist_robot2points([0,w],gpoints,corners)#h,0 for left hand or h and w*2 for right hand
        else:
            grasp_point,grasp_index,corner_point,corner_index = min_dist_robot2points([h*2,w],gpoints,corners)#h,0 for left hand or h and w*2 for right hand
        #grip_corners = corner_pairs[grasp_index+1]
        cv2.line(img, (pole[0] , pole[1]), (grasp_point[0], grasp_point[1]), (0,255,0),2)
        cv2.imshow("Corners",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        angle = self.alt_grip_angle(pole, grasp_point)
        print('Grasp point wo off' , grasp_point)
        print('BBB', bbox)
                
        # Get corner closest to grasp point

        # corner_point = grasp_point.copy()
        corner_point = self.get_mindis_point(grasp_point,tri_corners)
        
        # Translate points to full image frame
        grasp_point[0] = grasp_point[0]+bbox[2]
        grasp_point[1] = grasp_point[1]+bbox[0]
        corner_point[0] += bbox[2]
        corner_point[1] += bbox[0]
        print('Trinagle corner to go to ', corner_point)
        # Display grasp point
        cv2.circle(self.color_frame,(corner_point[0],corner_point[1]),2,(0,255,255),-1)    
        cv2.circle(self.color_frame,(grasp_point[0],grasp_point[1]),2,(0,255,0),-1)    

        cv2.imshow("Grasppoint", self.color_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

     

        # Convert corner and grasp point to robot space
        corner_point = np.array(corner_point)
        z_coord = get_max_depth(depth_ROI)
        yumi_corner_pose = cam2robot(corner_point[0], corner_point[1], z_coord , K,limb)
        yumi_corner_pose[2] = self.approach_height 

        grasp_point = np.array(grasp_point)
        print('Grasp point' , grasp_point)
        z_coord = get_min_depth(depth_ROI)
        yumi_grasp_pose = cam2robot(grasp_point[0], grasp_point[1],z_coord,K,limb)
        yumi_grasp_pose[2] = self.grasp_height
        print(yumi_grasp_pose)

        return yumi_grasp_pose,angle,yumi_corner_pose

   
    # Given an ROI in camera space of a triangle it
    # returns grasp corners  and angle in the camera space
    def peg_grasp_points(self,ROI,bbox,limb,K):
        if limb=="left":
            offset=self.left_grasp_offset
        elif limb=="right":
            offset=self.right_grasp_offset

        depth_ROI = self.depth_vals[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        gray = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        # dst = cv2.cornerHarris(gray,5,9,0.19)
        # dst = cv2.dilate(dst,None)
        # print dst.shape
        h,w = gray.shape
        h = h/2
        w = w/2
        cor_length = 0
        cv2.imshow("View_ROI",gray)
        cv2.waitKey(0)

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
        print('Cornersss', corners)
        cor_length = len(corners)
        if cor_length<3:
            return None, None, None
        pole = get_center_triangle(corners)
        gpoints = []
        corner_pairs = np.array([[[0,0],[0,0]]])
        # Get the actual grasping points, inside the object of interest
        for point1, point2 in list(itertools.combinations(corners, 2)):
            print('point', point1, point2)
            center = np.array([abs(point1[0]+point2[0])/2, abs(point1[1]+point2[1])/2])
            corner_pairs = np.concatenate((corner_pairs,np.array([[point1,point2]])),axis = 0)

            gpoint_plus = center + normal_to_line(point1,point2)*offset
            gpoint_minus = center - normal_to_line(point1,point2)*offset
            if np.linalg.norm(pole-abs(gpoint_plus))<np.linalg.norm(pole-abs(gpoint_minus)):
                gpoints.append(gpoint_plus.astype(int))

            else:
                gpoints.append(gpoint_minus.astype(int))

        gpoints = np.array(gpoints)
        print("Grasp Points",gpoints)

        for coords in gpoints:
            cv2.circle(ROI,(coords[0],coords[1]),1,(0,255,0),-1)
        cv2.imshow("Corners",ROI)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if limb == 'left':
            grasp_point,grasp_index,corner_point,corner_index = min_dist_robot2points([0,w],gpoints,corners)#h,0 for left hand or h and w*2 for right hand
        else:
            grasp_point,grasp_index,corner_point,corner_index = min_dist_robot2points([h*2,w],gpoints,corners)#h,0 for left hand or h and w*2 for right hand
        print(grasp_point, grasp_index, corner_point)
        grip_corners = corner_pairs[grasp_index+1]
        
        angle = self.grip_angle(grip_corners)
        print('Grasp point ori wo off' , grasp_point)
        print('BBB', bbox)

        grasp_point[0] = grasp_point[0]+bbox[2]
        grasp_point[1] = grasp_point[1]+bbox[0]
        cv2.circle(self.color_frame,(grasp_point[0],grasp_point[1]),3,(0,255,0),-1)
        # cv2.imshow("Grasppoint", self.color_frame)
        # cv2.waitKey(0)
        corner_point[0] = corner_point[0]+bbox[2]
        corner_point[1] = corner_point[1]+bbox[0]

        # Convert corner and grasp point to robot space
        corner_point = np.array(corner_point)
        z_coord = get_max_depth(depth_ROI)
        yumi_corner_pose = cam2robot(corner_point[0], corner_point[1], z_coord , K,limb)
        yumi_corner_pose[2] = self.approach_height 

        grasp_point = np.array(grasp_point)
        z_coord = get_min_depth(depth_ROI)
        yumi_grasp_pose = cam2robot(grasp_point[0], grasp_point[1],z_coord,K,limb)
        yumi_grasp_pose[2] = self.grasp_height
        print('Grasp point ori' , grasp_point)
        print(yumi_grasp_pose)
        return yumi_grasp_pose,angle,yumi_corner_pose
        return None, None, None


    # Given an ROI of the grasped peg  and  the pole positions in camera space,
    # return the drop points in robot space
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

        if limb=="right":
            center_coord_x = ((rob_pose.translation[0])+(corner_robot_pose[0]-rob_pose.translation[0])/3)
        else:
            center_coord_x = ((rob_pose.translation[0])+(corner_robot_pose[0]-rob_pose.translation[0])/3)
        center_coord_y = (rob_pose.translation[1])+(corner_robot_pose[1]-rob_pose.translation[1])/3

        print("gripper pose",rob_pose.translation[0],rob_pose.translation[1])
        print("corner pose",corner_robot_pose[0],corner_robot_pose[1])

        pole_pose = [pole_pos[0],pole_pos[1]]
        movetodelta=[pole_pose[0]-center_coord_x,pole_pose[1]-center_coord_y]
        print ("it will move to delta",movetodelta)
        # cv2.imshow("Drop_Corners",ROI)
        # cv2.waitKey(0)
        return movetodelta

    # same function as drop_pose but it does not rely on vision
    # the offsets are hardcoded
    def no_vision_drop_pose(self,ROI,limb,bbox,pole_pos):
        opposite_limb = 'right' if limb == 'left' else 'left'
        rob_pose = execution.get_curr_pose(opposite_limb)

        if limb=="right":
            center_coord_x = (rob_pose.translation[0])
            center_coord_y = (rob_pose.translation[1])
        else:
            center_coord_x = (rob_pose.translation[0])
            center_coord_y = (rob_pose.translation[1])

        print("gripper pose",rob_pose.translation[0],rob_pose.translation[1])
        # print("corner pose",corner_robot_pose[0],corner_robot_pose[1])

        pole_pose = [pole_pos[0],pole_pos[1]]
        movetodelta=[pole_pose[0]-center_coord_x+self.drop_offset[0],pole_pose[1]-center_coord_y+self.drop_offset[1]]
        print ("it will move to delta",movetodelta)
        # cv2.imshow("Drop_Corners",ROI)
        # cv2.waitKey(0)

        return movetodelta

    # Get all the pole positions in robot space in the
    # order defined in the experimental setup:
    # https://docs.google.com/spreadsheets/d/1UXPOGNIo5Zb-RXWs3y_bsV-bm5m3lPbPNlhAdudsr-I/edit?usp=sharing
    def pole_positions_rob(self,pole_poses,robot):
        if robot=="yumi":
            left_ids=[0,3,7,10,6,2]
            right_ids = [1,5,9,11,8,4]
        if robot=="taurus":
            left_ids=[0,3,7,10,6,2]
            right_ids = [1,5,9,11,8,4]

        poles_left = []
        poles_right = []
        poles_xyz=[]

        for x,y in pole_poses:
            cv2.circle(self.color_frame,(x,y),3,(0,255,0),-1)
            print('-'*10)
            print(x,y)
            poles_xyz.append([x,y,self.depth_vals[y,x]])

        poles_xyz = np.array(poles_xyz)
        for i in left_ids:
            poles_left.append(poles_xyz[i])
        poles_left = np.array(poles_left)
        self.left_poles_pixel_location=poles_left
        poles_left_arm = cam2robot_array(poles_left,self.K,limb)

        for i in right_ids:
            poles_right.append(poles_xyz[i])
        poles_right = np.array(poles_right)
        self.right_poles_pixel_location=poles_right
        poles_right_arm = cam2robot_array(poles_right,self.K,opposite_limb)

        #cv2.imshow("Poles",self.color_frame)
        # cv2.waitKey(0)
        return poles_left_arm,poles_right_arm

    # Given all the pegs, choose the closes to the selected pole
    # the format of the selected pole is an int that follows the
    # numbers of the experimental design
    # Returns the array index of the closest triangle
    def closest_ROI_to_pole(self,arm,selected_pole):
        triangles=self.pegs #triangles detected in order of confidence
        print(triangles)
        if arm=="left":
            poles_cam=self.left_poles_pixel_location
        if arm=="right":
            poles_cam=self.right_poles_pixel_location
        print("poles location shape",poles_cam.shape)
        selected_pole_XY=poles_cam[selected_pole-1,0:2]#selected pole is from 1-6, so -1 for 0-5
        # print("Selected pole",selected_pole_XY)

        # cv2.circle(self.color_frame,(int(selected_pole_XY[0]),int(selected_pole_XY[1])),3,(255,255,0),-1)
        # cv2.imshow("Rightt",self.color_frame)
        # cv2.waitKey(0)
        triangle_ids=range(len(triangles))
        closest_id=0
        dist=1000000000

        for i in triangle_ids:
            bbox=np.array(triangles[i])
            center_dist_2_pole=[(bbox[2]+bbox[3])/2-selected_pole_XY[0],(bbox[0]+bbox[1])/2-selected_pole_XY[1]]
            abs_dist=math.sqrt(center_dist_2_pole[0]**2+center_dist_2_pole[1]**2)
            if abs_dist<dist:
                dist=abs_dist
                closest_id=i
        print("dist",dist,"triangle id",closest_id)
        return(closest_id)

    # Given all the ROIs from the mask, choose the closest one
    # to the gripper.
    # Returns the array index of the closest triangle
    def closest_ROI_to_gripper(self):
        triangles=self.pegs #triangles detected in order of confidence
        print(triangles)
        triangle_ids=range(len(triangles))
        closest_id=0
        min_dist=1000000000

        for i in triangle_ids:
            bbox=np.array(triangles[i])
            depth_ROI = scene.depth_vals[bbox[0]:bbox[1],bbox[2]:bbox[3]]
            my_depth=get_average_depth(depth_ROI)

            if my_depth<min_dist:
                min_dist=my_depth
                closest_id=i

        print("dist",min_dist,"triangle id",closest_id)
        return(closest_id)

    # All ROS the subscribers
    def subscribe(self):
        rospy.init_node('PegAndPole', anonymous=True)
        # rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.get_bbox) #This is tthe subscriber for the darknet bounding boxes. SInce we use mask we dont need this
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb)
        rospy.Subscriber("/camera/color/camera_info",CameraInfo, self.camera_callback)
        rospy.Subscriber("/camera/color/image_raw",Image, self.image_callback)
        rospy.Subscriber("/masks_t",ObjMasks,self.mask_callback)
        rospy.Subscriber("/darknet_ros/tracked_bbs",TrackedBoundingBoxes,self.pole_cb)

######################################################################
############################# EXECUTION ##############################
######################################################################
if __name__ == '__main__':
    # Start Robot Environment
    # Start surgemes class and open grippers and take robot to neutral positions
    execution = Surgemes(strategy='model')
    time.sleep(1)
    # limb = input('Ente the limb : ')
    # limb = 'right'
    # opposite_limb = 'right' if limb == 'left' else 'left'


    # drop_pole_num = 1
    #################################### Initial setup complete

    # ROI_offset = 10 #ROI bounding box offsets
    # Start the Scene calss to obtain pole positions and bounding boxes etc.
    scene = Scene(execution)
    scene.subscribe()
    time.sleep(2)

    # DELETE 
    times_file = open("timestamps_execution.txt", "w")

    data_that_matters = []
    with open('Communication Message for Executiion - Taurus_sim.csv') as csvfile:
        robot="taurus"
        readCSV = csv.reader(csvfile)
        for row in readCSV:
            # data = np.array([row[3],row[4],row[7],row[10],row[8]])#grab start new peg transfer, prediction , pass_type , target_pole, intial peg poses 
            data = np.array([row[3],row[4],row[7],row[10],row[8]])
            data_that_matters.append(data)
    data_that_matters = np.array(data_that_matters)
    # print("grab start new peg transfer, prediction , pass_type , target_pole")
    data_that_matters = data_that_matters[1:,:]
    # print np.array(data_that_matters)

    input_data = []
    for x in data_that_matters:
        a = int(x[0])
        # print a
        b = int(x[1])
        if x[2] == 'L-R':
            x[2] = 'left'
        elif x[2] == 'R-L':
            x[2] = 'right'
        d = int(x[3])
        inp = [a,b,d]
        input_data.append(inp)
    input_data = np.array(input_data) #start nw tranfer, prediction, peg=dest
    # print(input_data)
    transfer_ids = np.where(input_data[:,0]==1)
    transfer_ids = np.array(transfer_ids)
    batches = []
    for x in transfer_ids[0,:]:
        # print x
        batch = input_data[x:x+7,1]
        batches.append(batch)
    # print batches
    #batches has the surgeme order for each transfer_id
    # find tranfer_id of input to get destination and l-r or r-l

    stop = 0
    surgeme_no = 0
    count = 0
    for x in transfer_ids[0,18:]:
        # print("Input Format: start_tranfer_index, prediction , pass_type , target_pole, order")
        # print("Input to be sent: ",data_that_matters[x,:])
        # print("Peg poses: ",data_that_matters[x,-1])
        # print("Limb: ",data_that_matters[x,2])
        limb = input("Enter the limb: ")
        opposite_limb = 'right' if limb == 'left' else 'left'
        rob_pose = execution.get_curr_pose(limb)
        time.sleep(0.5)
        execution.arm_close('left')
        time.sleep(0.5)
        execution.arm_open('left')
        time.sleep(0.5)
        execution.arm_close('right')
        time.sleep(0.5)
        execution.arm_open('right')
        time.sleep(0.5)

        # time.sleep(2)
        execution.ret_to_neutral_angles(limb)
        time.sleep(0.5)
        execution.ret_to_neutral_angles(opposite_limb)
        # time.sleep(2)
        while scene.pole_flag == 0:
            a = 1
        # print(scene.poles_found)
        left_poles,right_poles = scene.pole_positions_rob(scene.poles_found,robot) #Pole positions in robot space

        # print("Source and Dest: ",input_data[x,2])
        selected_pole = input("Enter source: ")
        selected_pole = int(selected_pole)
        # selected_pole = input_data[x,2]
        drop_pole_num = input("Enter Destination: ")
        drop_pole_num = int(drop_pole_num)-1
        # drop_pole_num = input_data[x,2]-1
        # sequence = batches[count]
        sequence = [0, 1, 2, 3, 4, 5, 6]
        print("Surgeme sequence: ",sequence)

        start_time = datetime.datetime.now()
        for surgeme in sequence:
            print("Surgeme: ",surgeme)
            # time.sleep(3)
            # surgeme_no = input('Enter the surgeme number youd like to perform: ')
            surgeme_no = int(surgeme)
            surgeme_no = surgeme_no+1

            if surgeme_no == 1:
                ############ SCENE SETUP ############################
                time.sleep(1)
                # selected_pole=input('Enter the destination pole : ')
                while len(scene.pegs) == 0:#wait for pegs to be deteced
                    print("Waiting for peg detection in surgeme 1")
                    a = 1
                selected_triangle_id=scene.closest_ROI_to_pole(limb,selected_pole)
                peg_coordinates = np.array(scene.pegs[selected_triangle_id]) #xmin,xmax,ymin,ymax choose peg closest to pole required 
                peg_coordinates = peg_coordinates.reshape(4)
                triangle_mask = scene.mask_frames[selected_triangle_id].copy()
                ROI = triangle_mask[peg_coordinates[0]:peg_coordinates[1],peg_coordinates[2]:peg_coordinates[3],:] #xmin,xmax,ymin,ymax
                ROI_temp = ROI.copy()
                ROI_corner = peg_coordinates.copy()
                grasp = None
                for ROI_offset in [0, 5, 10, 15, 20]:    
                    # selected_triangle_id=1
                    # print("esto",scene.pegs[int(selected_triangle_id)])
                    # peg_coordinates = np.array(scene.pegs[int(selected_triangle_id)]) #xmin,xmax,ymin,ymax choose peg closest to pole required 
                    # Apply offsets to pegs
                    peg_coordinates[0] = peg_coordinates[0]-ROI_offset
                    peg_coordinates[2] = peg_coordinates[2]-ROI_offset
                    peg_coordinates[1] = peg_coordinates[1]+ROI_offset
                    peg_coordinates[3] = peg_coordinates[3]+ROI_offset
                    print peg_coordinates

                    while len(scene.mask_frames) == 0:
                        print("waiting to detect poles")
                        a = 1
                    ROI = triangle_mask[peg_coordinates[0]:peg_coordinates[1],peg_coordinates[2]:peg_coordinates[3],:] #xmin,xmax,ymin,ymax
                    cv2.imshow("hello",ROI)
                    cv2.waitKey(0)
                    print(peg_coordinates)
                    depth_ROI = scene.depth_vals[peg_coordinates[0]:peg_coordinates[1],peg_coordinates[2]:peg_coordinates[3]]
                    grasp,g_angle,corner = scene.peg_grasp_points(ROI,peg_coordinates,limb,scene.K)# obtain corner points and grasp poiints from scene 
                    # if we found a grasp point we continue
                    if grasp is not None:
                        break
                # if grasp is None:

                cv2.imshow("New ROI",ROI_temp)
                cv2.waitKey(0)
                grasp,g_angle,corner = scene.alternate_grasp_points(ROI_temp,ROI_corner,limb,scene.K)# obtain corner points and grasp poiints from scene 
                ############ PERFORM APPROACH ############################
                execution.S1(corner,g_angle,limb)#Perform Approach
                print("Performed approach")
            if surgeme_no == 2:
                execution.S2(grasp,limb)#Perform Grasp
            if surgeme_no == 3:
                execution.S3(limb)#Perform Lift
            if surgeme_no == 4:
                execution.S4(limb)#Perform Go To transfer
            if surgeme_no == 5:
                transfer_flag = execution.S5(limb, opposite_limb)
            if surgeme_no == 6:
                # execution.y.left.goto_pose_delta([0,0,-0.03])
                # execution.y.right.goto_pose_delta([0,0,-0.03])
                time.sleep(1)
                # drop_pole_num = input('Enter the destination pole : ')-1 #please refer format
                # drop_pole_num = drop_pole_num-1
                if opposite_limb == 'right':
                    drop_pole_pose = right_poles[drop_pole_num]

                else:
                    drop_pole_pose = left_poles[drop_pole_num]

                while len(scene.pegs) == 0:
                    print("waiting to detect pegs")
                    a = 1

                drop_triangle_id=scene.closest_ROI_to_gripper()
                # selected_triangle_id=1

                peg_coordinates = []
                peg_coordinates = np.array(scene.pegs[drop_triangle_id]) #Choose peg closest to opposite limb gripper
                peg_coordinates = peg_coordinates.reshape(4)
                peg_coordinates[0] = peg_coordinates[0]-ROI_offset
                peg_coordinates[2] = peg_coordinates[2]-ROI_offset
                peg_coordinates[1] = peg_coordinates[1]+ROI_offset
                peg_coordinates[3] = peg_coordinates[3]+ROI_offset

                while len(scene.mask_frames) == 0:
                    print("waiting to detect masks for pegs")
                    a = 1

                transfer_flag = 0
                ROI = scene.mask_frames[drop_triangle_id][peg_coordinates[0]:peg_coordinates[1],peg_coordinates[2]:peg_coordinates[3],:] #xmin,xmax,ymin,ymax
                drop_pt = scene.no_vision_drop_pose(ROI,limb,peg_coordinates,drop_pole_pose)
                execution.S6(drop_pt,opposite_limb)#Perform Approach
            if surgeme_no == 7:
                execution.S7(opposite_limb,limb)#Perform Drop
                finish_time = datetime.datetime.now()
                time_diff = finish_time - start_time
                execution_time = time_diff.total_seconds() * 1000
                print("Finished trial", count, execution_time)
                times_file.write(str(execution_time)+"\n")
            # stop = input('Do you want to stop: ')
            time.sleep(0.5)
        count = count+1

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        times_file.close()
        exit(0)
    rospy.spin()


