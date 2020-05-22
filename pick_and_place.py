from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
from os import listdir
from os.path import isfile, join
from helpers import load_pose_by_path
from time import sleep
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
rospy.init_node("yumi")



data_path = 'data/pick_place'
#Play the trajectory
left_files = [f for f in listdir(data_path) if (isfile(join(data_path, f)) and ('left' in f))]
right_files = [f for f in listdir(data_path) if (isfile(join(data_path, f)) and ('right' in f))]
left_files.sort()
right_files.sort()
y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
pos = y.left.get_pose()


pub_l = rospy.Publisher('yumi_pose_left', Pose, queue_size=1)
pub_r = rospy.Publisher('yumi_pose_right', Pose, queue_size=1)
# pub_l = rospy.Publisher('yumi_pose_left', Pose, queue_size=1)
# pub_r = rospy.Publisher('yumi_pose_right', Pose, queue_size=1)

def pose_cb(data):
    pos = Pose()
    cpos = y.left.get_pose()
    pos.position.x = cpos.translation[0]
    pos.position.y = cpos.translation[1]
    pos.position.z = cpos.translation[2]
    pos.orientation.x = cpos.quaternion[0]
    pos.orientation.y = cpos.quaternion[1]
    pos.orientation.z = cpos.quaternion[2]
    pos.orientation.w = cpos.quaternion[3]
    pub_l.publish(pos)

    cpos = y.right.get_pose()
    pos.position.x = cpos.translation[0]
    pos.position.y = cpos.translation[1]
    pos.position.z = cpos.translation[2]
    pos.orientation.x = cpos.quaternion[0]
    pos.orientation.y = cpos.quaternion[1]
    pos.orientation.z = cpos.quaternion[2]
    pos.orientation.w = cpos.quaternion[3]
    pub_r.publish(pos)
rospy.Subscriber('yumisub', String, pose_cb)



#setup the ool distance for the surgical grippers
# ORIGINAL GRIPPER TRANSFORM IS tcp2=RigidTransform(translation=[0, 0, 0.156], rotation=[[ 1. 0. 0.] [ 0. 1. 0.] [ 0. 0. 1.]])
DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
y.left.set_tool(DELTALEFT)
y.right.set_tool(DELTARIGHT)
y.set_v(40)
y.set_z('z100')
y.left.move_gripper(0.005)
y.right.move_gripper(0.005)

# play the left files
count = 0
for l_file in left_files:
    l_pose = load_pose_by_path(join(data_path, l_file))
    pos.translation = l_pose['left'].translation
    pos.rotation = l_pose['left'].rotation
    if count ==2:
        pos.translation += [0,0,0.002]
    if count == 3:
        y.left.close_gripper(force=2,wait_for_res=False)
    y.left.goto_pose(pos,False,True,False)
    sleep(1)
    count += 1
sleep(1)

count = 0
for r_file in right_files:
    l_pose = load_pose_by_path(join(data_path, r_file))
    pos.translation = l_pose['right'].translation
    pos.rotation = l_pose['right'].rotation
    if count == 3:
        y.right.close_gripper(force=2,wait_for_res=False)
    y.right.goto_pose(pos,False,True,False)
    sleep(1)
    count += 1
sleep(3)
y.right.move_gripper(0.005)
y.left.move_gripper(0.005)
y.stop()



