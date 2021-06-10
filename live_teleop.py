import os, time, sys
import rospy
from sensor_msgs.msg import Image
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
import cv2
import numpy as np
import quaternion
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from autolab_core import RigidTransform
from helpers import load_pose_by_path
from collections import deque

sys.path.append('/home/isat/Forward/forward-simulation')
from hydra_capture import HydraIn
import argparse


class Teleop():
    #######################
    #        INIT         #
    #######################
    def __init__(self, delay, subject, trial, output_path):
        self.output_path = output_path
        self.bridge = CvBridge()
        self.color_frame = np.zeros([480,640,3],dtype=np.uint8)
        self.neutral_angles = load_pose_by_path('data/neutral_joints_straight.txt')

        # self.y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)

        self.h = HydraIn()
        self.h.calibrate()
    
        # Create videowriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_name = self.output_path+"S"+str(subject)+"T"+str(trial)+"_D"+str(delay)+"_RGB.avi"
        self.time_name = self.output_path+"S"+str(subject)+"T"+str(trial)+"_D"+str(delay)+"_timestamps.txt"
        if os.path.exists(self.video_name) or os.path.exists(self.time_name):
            print("Subject or trial Already recorded")
            exit(0)
        self.vout = cv2.VideoWriter(self.video_name,fourcc, 30.0, (1920,1080))
        self.t_stamps = open(self.time_name,'w')

        # Buffer 
        self.bufsiz = 1000
        self.imgbuf = deque(maxlen=self.bufsiz)
        self.imgbuf_t = deque(maxlen=self.bufsiz)
        self.conbuf_t = deque(maxlen=self.bufsiz)
        self.conbuf = deque(maxlen=self.bufsiz) 
        
        # Timimg
        self.init_t = time.time()
        self.delay_t = delay/1000.0  # in s
        self.maxrate = 20 # Max freq of loop in Hz

        rospy.init_node('robot_teleop',anonymous=True)
        rospy.Subscriber("/camera/color/image_raw",Image, self.image_callback)
        
        # Move to neutral angles
        # self.gripper_open(limb="left")
        # self.gripper_open(limb="right")
        # time.sleep(0.5)
        # self.ret_to_neutral_angles('left')
        # self.ret_to_neutral_angles('right')

        self.lpub = rospy.Publisher('ldiff',Float32MultiArray,queue_size=10)
        self.rpub = rospy.Publisher('rdiff',Float32MultiArray,queue_size=10)
        self.gpub = rospy.Publisher('grip',Float32MultiArray,queue_size=10)
        

    # def ret_to_neutral_angles(self,limb):
    #     self.y.set_v(80)
    #     limb_angles = 'left_angles' if limb == 'left' else 'right_angles'
    #     self.goto_joint_state(self.neutral_angles[limb_angles].joints,limb)
    #     time.sleep(1.0)

    # def goto_joint_state(self,joints,limb):
    #     arm = self.y.right if limb == 'right' else self.y.left
    #     joint_state = YuMiState(vals=joints)
    #     arm.goto_state(joint_state,False)       

    # def get_curr_pose(self,limb):
    #     arm = self.y.right if limb == 'right' else self.y.left
    #     return arm.get_pose()

    # def gripper_close(self, limb):
    #     arm = self.y.left if limb == 'left' else self.y.right
    #     arm.close_gripper(force=12, no_wait=True, wait_for_res=False)
    #     time.sleep(0.01)

    # def gripper_open(self, gripper_value=0.005, limb='right'):
    #     arm = self.y.left if limb == 'left' else self.y.right
    #     arm.move_gripper(gripper_value, no_wait=True, wait_for_res=False)
    #     time.sleep(0.01)

    
    def image_callback(self, data):
        color_frame = self.bridge.imgmsg_to_cv2(data, "rgb8")
        self.color_frame = cv2.cvtColor(color_frame,cv2.COLOR_BGR2RGB)
        self.imgbuf_t.append(time.time()-self.init_t)
        self.imgbuf.append(self.color_frame)
        self.vout.write(self.color_frame)
        self.t_stamps.write(str(time.time()-self.init_t)+'\n')
        # vision_logger.info('Received RGB Image from ROS')

    def loop(self):
        first_time = True
        l_diff_poses = np.zeros(7)
        r_diff_poses = np.zeros(7)


        r_closed = False
        l_closed = False
        k = ord('w')
        r = rospy.Rate(self.maxrate)
        showq = False
        img = np.zeros([480,640,3])
        while not rospy.is_shutdown():
            cur_t = time.time()-self.init_t
            
            # Display scene
            if len(self.imgbuf) > 0:
                # print('---',(cur_t-self.imgbuf_t[0]))
                while len(self.imgbuf) > 0 and (cur_t-self.imgbuf_t[0]) > self.delay_t:
                    img = np.copy(self.imgbuf.popleft())
                    self.imgbuf_t.popleft()

                
                # if (cur_t-self.imgbuf_t[0]) > self.delay_t:

                # #     # Pop image from buffer
                #     img = self.imgbuf.popleft()
                # #     # Clear time for that image
                #     self.imgbuf_t.popleft()

                img = cv2.flip(img,1)
                cv2.imshow('image',img)
                k=cv2.waitKey(1)


            if k==ord('q'):
                break

            

            if k==ord(' '):
                l_poses,r_poses,gripper_poses = self.h.get_vals_live()
                self.conbuf_t.append(time.time()-self.init_t)
                self.conbuf.append(np.concatenate((l_poses,r_poses,gripper_poses)))

            
            if len(self.conbuf) > 0 and ((cur_t-self.conbuf_t[0]) > self.delay_t):
                all_poses = self.conbuf.popleft()
                self.conbuf_t.popleft()

                l_poses = all_poses[:7]
                r_poses = all_poses[7:14]
                gripper_poses = all_poses[14:]

                if first_time:
                    l_temp_poses = l_poses
                    r_temp_poses = r_poses
                    first_time = False
                else:
                    l_diff_poses[:3] = l_poses[:3] -l_temp_poses[:3]
                    r_diff_poses[:3] = r_poses[:3] - r_temp_poses[:3]
                    
                    l_rot = np.quaternion(*l_poses[3:])*\
                            (np.quaternion(*l_temp_poses[3:]).inverse())
                    r_rot = np.quaternion(*r_poses[3:])*\
                            (np.quaternion(*r_temp_poses[3:]).inverse())

                    l_diff_poses[3:] = quaternion.as_float_array(l_rot)
                    r_diff_poses[3:] = quaternion.as_float_array(r_rot)
                    l_gripper_pose = 0.005*(1-gripper_poses[0])
                    r_gripper_pose = 0.005*(1-gripper_poses[1])


                    self.lpub.publish(Float32MultiArray(data=l_diff_poses))
                    self.rpub.publish(Float32MultiArray(data=r_diff_poses))
                    self.gpub.publish(Float32MultiArray(data=[l_gripper_pose, r_gripper_pose]))
            
            if k==ord(' '):            
                    #Update temp
                l_temp_poses = l_poses
                r_temp_poses = r_poses

            r.sleep()
            # print(time.time()-cur_t-self.init_t)

        cv2.destroyAllWindows()
        self.vout.release()
        print('Wrote Video')
        self.y.stop()
        self.t_stams.close()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action="store", dest="subject", type=int,
            help="Subject number")
    parser.add_argument('-t', action="store", dest="trial", type=int,
            help="Trial number")
    parser.add_argument('-d', action="store", dest="delay", type=int,
            help="Delay in milli seconds")
    parser.add_argument('-o', action="store", dest="output_path", 
            default="data/teleop_experiment/",
            help="Execution video and timestamp output paths")
    args = parser.parse_args()
    # TODO: Glebys will  add NICELY with HELP and EVERYTHING 
    t = Teleop(args.delay, args.subject, args.trial, args.output_path)
    t.loop()
    