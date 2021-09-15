import sys
import time
import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import PointCloud2 as msg_PointCloud2
import sensor_msgs.point_cloud2 as pc2_to_xyzrgb
from sensor_msgs.msg import Imu as msg_Imu
from sensor_msgs.msg import CameraInfo
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import inspect
import ctypes
import struct
import tf
from os.path import join
import pickle
from pprint import pprint as pp
import cv2
import pyrealsense2 as rs
import argparse


class realsense_calib():
    def __init__(self, robot):
        self.save_img=True
        self.count = 0
        self.bridge = CvBridge()
        self.cloud_frame =[]
        self.color_frame =[]
        self.depth_frame =[]
        self.P = []
        self.robot = robot
        if self.robot == 'taurus':
            self.img_path ='../taurus_control/data'
        else:
            self.img_path ='data'
        rospy.init_node('listener', anonymous=True)
        # rospy.Subscriber("/camera/aligned_depth_to_color/color/points",msg_PointCloud2, self.cloud_callback)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb)
        rospy.Subscriber("/camera/color/image_raw",Image, self.color_callback)
        rospy.Subscriber("/camera/color/camera_info",CameraInfo, self.info_callback)

        # rospy.Subscriber("/camera/color/camera_info",CameraInfo, self.camera_callback)

    def depth_cb(self,data):
        #img = data
        #rospy.loginfo(img.encoding)
        try:
                data.encoding = "mono16"
                cv_image = self.bridge.imgmsg_to_cv2(data, "mono16")
        except CvBridgeError as e:
                print(e)

        (rows,cols) = cv_image.shape
        self.depth_frame = cv_image/1000.0

    def save_imgs(self):
        cv2.imshow('img', self.color_frame)
        pixel_coord = []
        # if key is pressed save imgage
        keypress = cv2.waitKey(1)
        if ord('y') == keypress:
            # save the color image
            cv2.imwrite(join(self.img_path,'calibration_img'+str(self.count))+self.robot+".jpg",self.color_frame)
            # save the pointcloud
            np.save(join(self.img_path,'calibration_depth'+str(self.count))+self.robot,self.depth_frame)
            np.save(join(self.img_path,'intrinsics'+self.robot),self.K)
            # for points in self.cloud_frame:
                # pixel_coord.append(list(points[:3]) + self.point_2_pixel(points[:3]))
                # np.save(join(self.img_path,'pointcloud_'+str(self.count)), pixel_coord)
            self.count +=1
            print("saving image", self.count)

    def pc2_to_xyzrgb(self, point):
        # Thanks to Panos for his code used in this function.
        x, y, z = point[:3]
        rgb = point[3]
        s = struct.pack('>f', rgb)
        i = struct.unpack('>l', s)[0]
        # you can get back the float value by the inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)
        return x, y, z, r, g, b


    def cloud_callback(self, data):
            pass
        # print("yasss")
        # h,w = data.height, data.width
        # pts = np.frombuffer(data.data,dtype=np.float32)
                # print("cloud shape", h,w)
        # print("pts np shape:", pts.shape)

        # n = int(pts.shape[0]/(h*w))
        # pts = pts.reshape(h,w,n)
        # print(data.width)
        # print(data.fields)
        # print(data.fields.offset)
        # print(data.fields.count)
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        # self.cloud_frame = np.array([self.pc2_to_xyzrgb(pp) for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z", "rgb")) if pp[0] > 0])
        # print("XYZ",self.cloud_frame[1][:3])
        # print("RGB",self.cloud_frame[1][3:])
        # print(data.height)
        # print(data.width)

    def color_callback(self, data):
        # data.encoding = "bgr8"
        # color_intrin = data.profile.as_video_stream_profile().intrinsics
        self.color_frame = self.bridge.imgmsg_to_cv2(data, "rgb8")
        self.color_frame = cv2.cvtColor(self.color_frame,cv2.COLOR_BGR2RGB)
        # self.color_frame = cv2.asanarray(color_frame)
        self.save_imgs()
        # print('Image', data.height, data.width)

    def info_callback(self, data):

        # print(data.K)
        # data.encoding = "bgr8"
        self.K = np.array(list(data.K)).reshape(3,3)

    def camera_callback(self, data):
        # data.encoding = "bgr8"
        self.P = data.P


    def point_2_pixel(self, temp_cloud ):
        temp_cloud = list(temp_cloud) + [1]
        pix = np.dot( np.reshape(list(self.P),(3,4)),temp_cloud)
        pix = pix/pix[-1]
        return list(pix[:2])

if __name__ == '__main__':
    parser = parser = argparse.ArgumentParser()
    parser.add_argument('-r',dest='robot', help="robot")
    args = parser.parse_args()
    robot = "taurus" if args.robot=="taurus" else ""
    real_c = realsense_calib(robot)
    print("Press Y to save the images for calibration")
    while not rospy.is_shutdown():
        pass

# class CWaitForMessage:
#     def __init__(self, params={}):
#         self.result = None

#         self.break_timeout = False
#         self.timeout = params.get('timeout_secs', -1)
#         self.seq = params.get('seq', -1)
#         self.time = params.get('time', None)
#         self.node_name = params.get('node_name', 'rs2_listener')
#         self.bridge = CvBridge()
#         self.listener = None

#         self.themes = {'depthStream': {'topic': '/camera/depth/image_rect_raw', 'callback': self.imageColorCallback, 'msg_type': msg_Image},
#                        'colorStream': {'topic': '/camera/color/image_raw', 'callback': self.imageColorCallback, 'msg_type': msg_Image},
#                        'pointscloud': {'topic': '/camera/depth/color/points', 'callback': self.pointscloudCallback, 'msg_type': msg_PointCloud2},
#                        'alignedDepthInfra1': {'topic': '/camera/aligned_depth_to_infra1/image_raw', 'callback': self.imageColorCallback, 'msg_type': msg_Image},
#                        'alignedDepthColor': {'topic': '/camera/aligned_depth_to_color/image_raw', 'callback': self.imageColorCallback, 'msg_type': msg_Image},
#                        'static_tf': {'topic': '/camera/color/image_raw', 'callback': self.imageColorCallback, 'msg_type': msg_Image},
#                        'accelStream': {'topic': '/camera/accel/sample', 'callback': self.imuCallback, 'msg_type': msg_Imu},
#                        }

#         self.func_data = dict()

#     def imuCallback(self, theme_name):
#         def _imuCallback(data):
#             if self.listener is None:
#                 self.listener = tf.TransformListener()
#             self.prev_time = time.time()
#             self.func_data[theme_name].setdefault('value', [])
#             self.func_data[theme_name].setdefault('ros_value', [])
#             try:
#                 frame_id = data.header.frame_id
#                 value = data.linear_acceleration

#                 (trans,rot) = self.listener.lookupTransform('/camera_link', frame_id, rospy.Time(0))
#                 quat = tf.transformations.quaternion_matrix(rot)
#                 point = np.matrix([value.x, value.y, value.z, 1], dtype='float32')
#                 point.resize((4, 1))
#                 rotated = quat*point
#                 rotated.resize(1,4)
#                 rotated = np.array(rotated)[0][:3]
#             except Exception as e:
#                 print(e)
#                 return
#             self.func_data[theme_name]['value'].append(value)
#             self.func_data[theme_name]['ros_value'].append(rotated)
#         return _imuCallback            

#     def imageColorCallback(self, theme_name):
#         def _imageColorCallback(data):
#             self.prev_time = time.time()
#       rospy.loginfo('WHYYYY GOD WHYYYYYYYYYYYYY')
#             self.func_data[theme_name].setdefault('avg', [])
#             self.func_data[theme_name].setdefault('ok_percent', [])
#             self.func_data[theme_name].setdefault('num_channels', [])
#             self.func_data[theme_name].setdefault('shape', [])
#             self.func_data[theme_name].setdefault('reported_size', [])

#             try:
#                 cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
#             except CvBridgeError as e:
#                 print(e)
#                 return
#             channels = cv_image.shape[2] if len(cv_image.shape) > 2 else 1
#             pyimg = np.asarray(cv_image)

#             ok_number = (pyimg != 0).sum()
#             cv2.imshow('image',cv_image)

#             self.func_data[theme_name]['avg'].append(pyimg.sum() / ok_number)
#             self.func_data[theme_name]['ok_percent'].append(float(ok_number) / (pyimg.shape[0] * pyimg.shape[1]) / channels)
#             self.func_data[theme_name]['num_channels'].append(channels)
#             self.func_data[theme_name]['shape'].append(cv_image.shape)
#             self.func_data[theme_name]['reported_size'].append((data.width, data.height, data.step))
#         return _imageColorCallback

#     def imageDepthCallback(self, data):
#         pass

#     def pointscloudCallback(self, theme_name):
#         def _pointscloudCallback(data):
#             self.prev_time = time.time()
#             print 'Got pointcloud: %d, %d' % (data.width, data.height)

#             self.func_data[theme_name].setdefault('frame_counter', 0)
#             self.func_data[theme_name].setdefault('avg', [])
#             self.func_data[theme_name].setdefault('size', [])
#             self.func_data[theme_name].setdefault('width', [])
#             self.func_data[theme_name].setdefault('height', [])
#             # until parsing pointcloud is done in real time, I'll use only the first frame.
#             self.func_data[theme_name]['frame_counter'] += 1

#             if self.func_data[theme_name]['frame_counter'] == 1:
#                 # Known issue - 1st pointcloud published has invalid texture. Skip 1st frame.
#                 return

#             try:
#                 points = np.array([pc2_to_xyzrgb(pp) for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z", "rgb")) if pp[0] > 0])
#             except Exception as e:
#                 print(e)
#                 return
#             self.func_data[theme_name]['avg'].append(points.mean(0))
#             self.func_data[theme_name]['size'].append(len(points))
#             self.func_data[theme_name]['width'].append(data.width)
#             self.func_data[theme_name]['height'].append(data.height)
#         return _pointscloudCallback

#     def wait_for_message(self, params):
#         topic = params['topic']
#         print 'connect to ROS with name: %s' % self.node_name
#         rospy.init_node(self.node_name, anonymous=True)

#         rospy.loginfo('Subscribing on topic: %s' % topic)
#         self.sub = rospy.Subscriber(topic, msg_Image, self.callback)

#         self.prev_time = time.time()
#         break_timeout = False
#         while not any([rospy.core.is_shutdown(), break_timeout, self.result]):
#             rospy.rostime.wallsleep(0.5)
#             if self.timeout > 0 and time.time() - self.prev_time > self.timeout:
#                 break_timeout = True
#                 self.sub.unregister()

#         return self.result

#     @staticmethod
#     def unregister_all(registers):
#         for test_name in registers:
#             rospy.loginfo('Un-Subscribing test %s' % test_name)
#             registers[test_name]['sub'].unregister()

#     def wait_for_messages(self, themes):
#         # tests_params = {<name>: {'callback', 'topic', 'msg_type', 'internal_params'}}
#         self.func_data = dict([[theme_name, {}] for theme_name in themes])

#         print 'connect to ROS with name: %s' % self.node_name
#         rospy.init_node(self.node_name, anonymous=True)
#         for theme_name in themes:
#             theme = self.themes[theme_name]
#             rospy.loginfo('Subscribing %s on topic: %s' % (theme_name, theme['topic']))
#             self.func_data[theme_name]['sub'] = rospy.Subscriber(theme['topic'], theme['msg_type'], theme['callback'](theme_name))

#         self.prev_time = time.time()
#         break_timeout = False
#         while not any([rospy.core.is_shutdown(), break_timeout]):
#             rospy.rostime.wallsleep(0.5)
#             if self.timeout > 0 and time.time() - self.prev_time > self.timeout:
#                 break_timeout = True
#                 self.unregister_all(self.func_data)

#         return self.func_data

#     def callback(self, data):
#         rospy.loginfo('Got message. Seq %d, secs: %d, nsecs: %d' % (data.header.seq, data.header.stamp.secs, data.header.stamp.nsecs))

#         self.prev_time = time.time()
#         if any([self.seq > 0 and data.header.seq >= self.seq,
#                 self.time and data.header.stamp.secs == self.time['secs'] and data.header.stamp.nsecs == self.time['nsecs']]):
#             self.result = data
#             # self.sub.unregister()



# def main():
#     if len(sys.argv) < 2 or '--help' in sys.argv or '/?' in sys.argv:
#         print 'USAGE:'
#         print '------'
#         print 'rs2_listener.py <topic | theme> [Options]'
#         print 'example: rs2_listener.py /camera/color/image_raw --time 1532423022.044515610 --timeout 3'
#         print 'example: rs2_listener.py pointscloud'
#         print ''
#         print 'Application subscribes on <topic>, wait for the first message matching [Options].'
#         print 'When found, prints the timestamp.'
#         print
#         print '[Options:]'
#         print '-s <sequential number>'
#         print '--time <secs.nsecs>'
#         print '--timeout <secs>'
#         exit(-1)

#     # wanted_topic = '/device_0/sensor_0/Depth_0/image/data'
#     # wanted_seq = 58250

#     wanted_topic = sys.argv[1]
#     msg_params = {}
#     for idx in range(2, len(sys.argv)):
#         if sys.argv[idx] == '-s':
#             msg_params['seq'] = int(sys.argv[idx + 1])
#         if sys.argv[idx] == '--time':
#             msg_params['time'] = dict(zip(['secs', 'nsecs'], [int(part) for part in sys.argv[idx + 1].split('.')]))
#         if sys.argv[idx] == '--timeout':
#             msg_params['timeout_secs'] = int(sys.argv[idx + 1])

#     msg_retriever = CWaitForMessage(msg_params)
#     if '/' in wanted_topic:
#         msg_params = {'topic': wanted_topic}
#         res = msg_retriever.wait_for_message(msg_params)
#         rospy.loginfo('Got message: %s' % res.header)
#     else:
#         themes = [wanted_topic]
#         res = msg_retriever.wait_for_messages(themes)
#         print res


# if __name__ == '__main__':
#     main()

