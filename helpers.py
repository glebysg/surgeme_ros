# helper.py
# Contains miscelaneous helper functions
import numpy as np
import csv
import ast
import os
from os.path import isfile, join
from autolab_core import RigidTransform
from scipy import interpolate
import pickle as pkl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

################################
# Data parser functionj
# inputs:
    # data_file: if its a file, just process that file.
          # in that directory
    # param_file: task parameters path file
    # recursive: if True, process every subdirectory
               # inside the directory
    # arm: the arm of the robot that will be processed.
# output:
    # a n by m numpy array, where n is the sample number
    # and m is made of [tayectory_features (12), label,
    # target peg num, rotation type]
    # if the surgeme does not have a target peg/pole
    # (the case for get-together/exchagnge) the number
    # will be filled with zero.
    #################################################
    # Peg numbers are annotated like this (For Yumi):
    # (Human Left/Robot Right)    (Human Right/Robot Left)
          # 10                               4
      # 9         11                     3         5
      # 8         12                     2         6
           # 7                                1
H = np.loadtxt("./data/homography.txt")
def camera_to_world(point):
    global H
    point = np.array(point)
    estimated_dest = np.dot(H[:3,:],point)
    return estimated_dest

def parse_input_data(data_file, param_file, recursive=False, arm='left'):
    output = []
    # Get the data in the params file
    with open(param_file) as csv_param:
        param_reader = csv.reader(csv_param, delimiter=',')
        param_rows = [row for row in param_reader]

    file_list = []
    if os.path.isfile(data_file):
       file_list.append(data_file)
    elif recursive:
        for dir_name, _, _ in os.walk(data_file):
            dir_files = [f for f in os.listdir(dir_name) if isfile(join(dir_name, f))]
            file_list += [join(dir_name,f) for f in dir_files if "kinematics" in f \
                    and arm in f and f.endswith(".txt")]
    else:
        dir_files = [f for f in os.listdir(data_file) if isfile(join(data_file, f))]
        file_list = [join(data_file,f) for f in dir_files if  "kinematics" in f \
                    and arm in f and f.endswith(".txt")]
    file_list = sorted(file_list)
    for file_name in file_list:
        with open(file_name) as csv_data:
            data_reader = csv.reader(csv_data, delimiter=',')
            transfer_step_complete = False
            peg_index = 0
            # Filter out the param for the task
            task_params = None
            for first in data_reader:
                print file_name
                subject,trial = first[0].split("_")
                task_params = [p for p in  param_rows if \
                        p[0]==subject and p[1]==trial][0]
                task_params[3] = ast.literal_eval(str(task_params[3]))
                break
            # reset csv so we can look at it from the begining
            csv_data.seek(0)
            for data_row in data_reader:
                output_row = []
                # if there is no label, skip reading
                if data_row[22] == 'False':
                    continue
                # get the traslation, rotation
                output_row += map(float,data_row[9:21])
                # add the label as an int
                output_row.append(int(data_row[23][1]))
                # Check if current surgeme is S1, if S7 happened,
                # add 1 to the counter.  and reset the S7 checker.
                current_surgeme = output_row[-1]
                if current_surgeme == 1 and transfer_step_complete:
                    peg_index += 1
                    transfer_step_complete = False
                elif current_surgeme == 7:
                    transfer_step_complete = True
                # From surgemes 1-3 Fill it with the first peg-number
                # for 4-5 fill it with zeros. For 6-5 fill it with
                # the mirror peg on the other side.
                if current_surgeme < 4:
                    peg_offset = 6 if task_params[2] == 'L-R' else 0
                    peg_num = task_params[3][peg_index]+peg_offset
                elif current_surgeme < 6:
                    peg_num = 0
                else:
                    peg_offset = 0 if task_params[2] == 'L-R' else 6
                    peg_num = task_params[3][peg_index]+peg_offset
                output_row.append(peg_num)
                output_row.append(int(task_params[-1]))
                output.append(output_row)
    return np.array(output,dtype=float)

# Loads a pose object from a file
# inputs:
    # pose_path: path to the .pkl file of the pose
# output:
    # the pose object
def load_pose_by_path(pose_path):
    with open(pose_path, 'rb') as file_name:
        pose = pkl.load(file_name)
        return pose

# Loads a pose object from the pose description
# inputs:
    # arm: robot arm it accepts strings and only
        # three options: 'left' , 'right' or 'both'
    # peg: the peg number where the arm is located
    # rot: the current rotation type of the pegboard.
# output:
    # the pose object
def load_pose_by_desc(arm='left',peg=1,rot=1):
    with open('./poses/'+arm+'_'+str(peg)+'_'+str(rot)+'_pose', 'rb') as file_name:
        pose = pkl.load(file_name)
        return pose

# given a 3d trajectory, returns the spline params
# inputs:
    # trayectory: nx3 array of points. each column
    # a dimension.
# output:
    # the spline params tck and order see
    # scipy.interpolate.BSpline for the exact
    # parameter structure
def get_spline(trajectory, s=3):
    _,x_index = np.unique(trajectory[:,0],return_index = True)
    _,y_index = np.unique(trajectory[:,1],return_index = True)
    _,z_index = np.unique(trajectory[:,2],return_index = True)
    xyz_index = np.concatenate((x_index,y_index,z_index),axis = None)
    xyz_index = np.sort(np.unique(xyz_index))

    unique_pos = trajectory[xyz_index]
    tck, u = interpolate.splprep([
        unique_pos[:,0],
        unique_pos[:,1],
        unique_pos[:,2]], s=s)
    return tck, u

def get_waypoints(trajectory, s=3):
    _,x_index = np.unique(trajectory[:,0],return_index = True)
    _,y_index = np.unique(trajectory[:,1],return_index = True)
    _,z_index = np.unique(trajectory[:,2],return_index = True)
    xyz_index = np.concatenate((x_index,y_index,z_index),axis = None)
    xyz_index = np.sort(np.unique(xyz_index))

    unique_pos = trajectory[xyz_index]
    # if the trajectory has less than 6 datapoints,
    # do not take it into account for training
    if unique_pos.shape[0] < 6:
        return None
    tck, u = interpolate.splprep([
        unique_pos[:,0],
        unique_pos[:,1],
        unique_pos[:,2]], s=s)

    num_t = 30
    t_points = np.linspace(0,1,num_t)
    x_pred, y_pred, z_pred = interpolate.splev(t_points, tck)
    # Add the initial point to the spline
    # Add midway knowt points to the spline
    t_step = num_t/s
    way_points = []
    for i in range(1,s):
        way_points.append(x_pred[i*t_step])
        way_points.append(y_pred[i*t_step])
        way_points.append(z_pred[i*t_step])
    # x_3.append(unique_pos[-1,0])
    # y_3.append(unique_pos[-1,1])
    # z_3.append(unique_pos[-1,2])
    # tck_3, u_3 = interpolate.splprep([x_3,y_3,z_3 ], s=s)
    # x_pred_3, y_pred_3, z_pred_3 = interpolate.splev(t_points, tck_3)

    # fig = plt.figure(2)
    # ax3d = fig.add_subplot(111, projection='3d')
    # ax3d.plot(unique_pos[:,0], unique_pos[:,1],  unique_pos[:,2], 'b')
    # ax3d.plot(x_pred, y_pred, z_pred, 'r')
    # ax3d.plot(x_pred_3, y_pred_3, z_pred_3, 'g')
    # ax3d.scatter(x_3, y_3, z_3, 'go')
    # fig.show()
    # plt.show()
    return way_points

def get_3dpt_depth(pixel, depth, k):
    cx = k[0,2]
    cy = k[1,2]
    fx = k[0,0]
    fy = k[1,1]
    u = pixel[0]
    v = pixel[1]
    x = (u - cx)*depth/fx
    y = (v - cy)*depth/fy
    return np.array([x,y,depth])