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
import argparse
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# example run
# predict surgeme 3 with the left arm using Linear Regression
# python modeled_spline_prediction.py -n 3 -a left -m LR
##########################
###     PARAMS         ###
##########################
parser = argparse.ArgumentParser()
parser.add_argument('-n', action="store", dest="surgeme_number", default=1,
        type=int, help="number of the surgeme to predict")
parser.add_argument('-d', action="store", dest="spline_degree", default=3,
        type=int, help="degree of the spline that will be predicted")
parser.add_argument('-b', action="store", dest="base_dir",
        default="./data/Yumi/",
        help="base directory where all the subject data collection folders are")
parser.add_argument('-p', action="store", dest="param_file",
        default="./data/Yumi/task_params.csv",
        help="parameter file describing the rotation and pegs for each trial")
parser.add_argument('-a', action="store", dest="arm", default="left",
        help="Robot arm to predict. Possible options: left and right")
parser.add_argument('-m', action="store", dest="model", default="RL",
        help="Model to use for prediction. Possible options: NN or RL")
parser.add_argument('--save_img',action="store_true", default=False,
        help="Save the predicted figures")
args = parser.parse_args()
# Load the data for the left arm
save_model = False if args.save_img is None else True
arm = args.arm
arm_data = parse_input_data(args.base_dir,args.param_file,True,arm)
surgeme_number = args.surgeme_number
# Spline Params
spline_degree = args.spline_degree
coeff_len = (spline_degree - 1)*3
print("COEFF LEN", coeff_len)

# Test params
NN = True if args.model == 'NN' else False# use NN or Regresion

# Filter it to surgeme 1 that stats with the left arm pickup
if surgeme_number == 4 or surgeme_number == 5:
    s_data = arm_data[(arm_data[:,12]==surgeme_number)&(arm_data[:,13]==0)]
    pose_index = []
elif arm == "right":
    s_data = arm_data[(arm_data[:,12]==surgeme_number)&(arm_data[:,13]>6)]
    pose_index = range(7,13)
else:
    s_data = arm_data[(arm_data[:,12]==surgeme_number)&(arm_data[:,13]<7)]
    pose_index = range(1,7)
print s_data.shape
# Rebase by target
# subtract the n peg by with rotation x with peg 2 with rotation 1 (base peg)
# Load Peg positions
pose_matrix = []
for rot in range(1,4):
    rot_poses = []
    for peg in pose_index:
        pos = load_pose_by_desc(arm,peg,rot)
        rot_poses.append(pos)
    pose_matrix.append(rot_poses)
# finised Peg position loading
# This vector is then subtracted to all the points in the sample.
# and saved in rebased data. Rebased data saves all the data segments

rebased_data = []
surgeme = []
prev_peg = -1
prev_rot = -1
first = True
for elem in s_data:
    row = []
    # get the rebasing translation
    elem_rot = int(elem[14])
    elem_peg = int(elem[13])
    if (elem_peg != prev_peg or elem_rot != prev_rot) and not first:
        # surgeme finished
        rebased_data.append(surgeme)
        surgeme = []
    if surgeme_number == 4 or surgeme_number == 5:
        rebased_pos = elem[0:3]
    else:
        if arm == "right":
            offset = pose_matrix[elem_rot-1][elem_peg-7][arm].translation-pose_matrix[0][0][arm].translation
        else:
            offset = pose_matrix[elem_rot-1][elem_peg-1][arm].translation-pose_matrix[0][0][arm].translation
        rebased_pos = elem[0:3]-offset
    row += list(rebased_pos)
    row += list(elem[12:])
    surgeme.append(row)
    prev_peg = elem_peg
    prev_rot = elem_rot
    first = False
print(np.array(rebased_data).shape)
#For each surgeme, get the splines (keep target peg, label and rotation)
final_data = []
rebased_index = 0
for elem in rebased_data:
    surgeme = np.array(elem)
    surgeme_pos = surgeme[:,:3]
    # get the Middle waypoitns from the spline
    way_points = get_waypoints(surgeme_pos)
    if way_points is None:
        rebased_index += 1
        continue
    # Create a new data point
    data_row = []
    # INPUT DATA
    # add start point
    data_row.extend(list(surgeme[0,:3]))
    # add target
    data_row.extend(list(surgeme[-1,:3]))
    # OUTPUT DATA
    # add the 2 waypoints to be predicted (6 numbers, 2 3D points)
    data_row.extend(way_points)
    # EXTRA PLOTTING INFO
    # add the label, peg and rotation
    data_row.extend(list(surgeme[0,3:]))
    # add the index of the original data
    data_row.append(rebased_index)
    final_data.append(data_row)
    rebased_index += 1
final_data = np.array(final_data)

# Train
# get training and testing splits
x_full = final_data[:,:6]
y_full = final_data[:,6:]
x_train, x_test, y_train, y_test = train_test_split(
    x_full, y_full, test_size=0.2, random_state=42)
# y_train = y_train.reshape((-1,1))
# y_test = y_test.reshape((-1,1))
reg = LinearRegression().fit(x_train, y_train[:,:coeff_len])
print "REGRESSION SCORE", reg.score(x_train, y_train[:,:coeff_len])
if save_model:
    with open('models/S'+str(surgeme_number)+"_"+arm+'_regression', 'wb') as model_name:
        pkl.dump(reg,model_name)

# TRY WITH A NN_REGRESSION
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,15)
        self.fc2 = nn.Linear(15,15)
        self.fc3 = nn.Linear(15,coeff_len)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()

# Normalization and preprocessing here
# Make the data into tensors
train_tensor = torch.tensor(np.concatenate((x_train,y_train), axis=1)).float()
test_tensor = torch.tensor(np.concatenate((x_test,y_test), axis=1)).float()

for epoch in range(40):
    for i, data2 in enumerate(train_tensor):
        X=data2[0:6]
        Y=data2[6:6+coeff_len]
        X, Y = Variable(X, requires_grad=True), Variable(Y, requires_grad=False)
        optimizer.zero_grad()
        y_pred = net(X)
        output = criterion(y_pred, Y)
        output.backward()
        optimizer.step()
    if (epoch % 20 == 0.0):
        print("Epoch {} - loss: {}".format(epoch, output))

torch.save(net,'models/S'+str(surgeme_number)+"_"+arm+'_nn')

###############################################################
###################### testing ################################
###############################################################
# for each element in the testing
# plot the real curve, the target curve and the predicted curve
count = 1
dtw_measurements = []
for data, target in zip(x_test,y_test):
    title = "NN" if NN else "LR"
    title += " sample " + str(count)
    count += 1
    # data = [surgeme start, surgeme target]
    # target = [spline coefficients,spline_knots,peg,rotation,label,rebased_index]

    # Get the original x,y,z
    t_points = np.linspace(0,1,30)
    rebased_index = target[-1]
    orig_data = np.array(rebased_data[int(rebased_index)])
    x_orig = orig_data[:,0]
    y_orig = orig_data[:,1]
    z_orig = orig_data[:,2]

    # Get the target spline
    t_points = np.linspace(0,1,30)
    target_tck, target_u = get_spline(orig_data)
    x_target, y_target, z_target = interpolate.splev(t_points, target_tck)

    # Get the predicted spline
    if NN:
        tensor_data= torch.tensor(data).float()
        tensor_target= torch.tensor(target[:coeff_len]).float()
        X, Y = Variable(tensor_data, requires_grad=False), Variable(tensor_target, requires_grad=False)
        pred_waypoints = net(X)
        loss = criterion(pred_waypoints, Y)
        pred_waypoints = net(X).data.numpy()
        print "MSE", loss
    else:
        print data
        pred_waypoints = reg.predict(data.reshape(1,-1))
        print "PREDICTON", pred_waypoints

    # Get the predicted spline
    # add the initial point
    x_way = []
    y_way = []
    z_way = []
    x_way.append(orig_data[0,0])
    y_way.append(orig_data[0,1])
    z_way.append(orig_data[0,2])
    # add the waypoints
    pred_waypoints = pred_waypoints.reshape(coeff_len)
    for i in range(coeff_len/3):
        x_way.append(pred_waypoints[i*3])
        y_way.append(pred_waypoints[i*3 +1])
        z_way.append(pred_waypoints[i*3 +2])
    # add the endpoint
    x_way.append(orig_data[-1,0])
    y_way.append(orig_data[-1,1])
    z_way.append(orig_data[-1,2])
    # get the spline
    t_points = np.linspace(0,1,30)
    tck_way, u_way = interpolate.splprep([x_way,y_way,z_way ], s=spline_degree)
    x_pred, y_pred, z_pred = interpolate.splev(t_points, tck_way)

    ##########################################
    ########### get the DTW distance #########
    ##########################################
    target_curve = zip(x_target,y_target,z_target)
    predicted_curve = zip(x_pred,y_pred,z_pred)
    dtw_dist, _ = fastdtw(target_curve, predicted_curve, dist=euclidean)
    dtw_measurements.append(dtw_dist)
    ##########################################

    fig = plt.figure(2)
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(x_orig, y_orig, z_orig, 'b')
    ax3d.plot(x_target, y_target, z_target, 'r')
    ax3d.plot(x_pred, y_pred, z_pred, 'go')
    plt.title(title)
    if args.save_img:
        fig.savefig('data/S'+str(surgeme_number)+"_"+title+'.png')
    else:
        plt.show()
print("DTW MEAN:", np.mean(dtw_measurements), "DTW STD", np.std(dtw_measurements))
# Print the DTW distance with the real trajectory and the target curve
# [array([0., 0., 0., 0., 1., 1., 1., 1.]), [array([0.48237748, 0.47691105, 0.44821935, 0.44967256]), array([0.08779603, 0.10004131, 0.10692803, 0.10645012]), array([0.07841477, 0.05831569, 0.03754693, 0.01694974])], 3]
