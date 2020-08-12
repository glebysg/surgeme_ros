from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
from helpers import load_pose_by_path
from time import sleep

limb = 'right'

neutral_pose = load_pose_by_path('data/neutral_pose_peg.txt')
neutral_angles = load_pose_by_path('data/neutral_angles_pose_peg1.txt')
transfer_pose_high_left = load_pose_by_path('data/transfer_pose_high.txt')
transfer_pose_low_left = load_pose_by_path('data/new_transfer_pose_low_left_to_right.txt')
transfer_pose_high_right = load_pose_by_path('data/transfer_pos_high_right_to_left.txt')
transfer_pose_low_right = load_pose_by_path('data/new_transfer_pose_low_right_to_left.txt')

y = YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
y.left.set_tool(DELTALEFT)
y.right.set_tool(DELTARIGHT)
y.set_v(80)
y.set_z('z100')

y.left.move_gripper(0.005)
y.right.move_gripper(0.005)

arm = y.right if limb == 'right' else y.left
opposite_arm = y.left if limb == 'right' else y.right
limb_angles = 'left_angles' if limb == 'left' else 'right_angles'
opposite_angles = 'right_angles' if limb == 'left' else 'left_angles'

# Move to Neutral
curr_pos_limb = arm.get_state()
des_pos_limb = curr_pos_limb
des_pos_limb.joints = neutral_angles[limb_angles].joints
arm.goto_state(des_pos_limb)

curr_pos_limb = opposite_arm.get_state()
des_pos_limb = curr_pos_limb
des_pos_limb.joints = neutral_angles[opposite_angles].joints
opposite_arm.goto_state(des_pos_limb)

sleep(3)

# Close the main limb
if limb == "left":
    y.left.close_gripper(force=12,wait_for_res=False)
else:
    y.right.close_gripper(force=12,wait_for_res=False)

# Get together
curr_pos = arm.get_pose()
# print "Current location after returning to neutral: ", curr_pos
# print "Shuting yumi"
# self.y.stop()
opposite_arm.move_gripper(0.005)
curr_pos_limb = arm.get_state()
des_pos_limb = curr_pos_limb
if limb == 'left':
        des_pos_limb.joints = transfer_pose_high_left[limb_angles].joints
else:
        des_pos_limb.joints = transfer_pose_high_right[limb_angles].joints
arm.goto_state(des_pos_limb)

curr_pos_opposite_limb = opposite_arm.get_state()
des_pos_opposite_limb = curr_pos_opposite_limb

if limb == 'left':
        des_pos_opposite_limb.joints = transfer_pose_high_left[opposite_angles].joints
else:
        des_pos_opposite_limb.joints = transfer_pose_high_right[opposite_angles].joints

opposite_arm.goto_state(des_pos_opposite_limb)

sleep(1)
################################ DO Transer pose lose #######

curr_pos_limb = arm.get_state()
des_pos_limb = curr_pos_limb
if limb == 'left':
        des_pos_limb.joints = transfer_pose_low_left[limb_angles].joints
else:
        des_pos_limb.joints = transfer_pose_low_right[limb_angles].joints
arm.goto_state(des_pos_limb)

curr_pos_opposite_limb = opposite_arm.get_state()
des_pos_opposite_limb = curr_pos_opposite_limb
if limb == 'left':
        des_pos_opposite_limb.joints = transfer_pose_low_left[opposite_angles].joints
else:
        des_pos_opposite_limb.joints = transfer_pose_low_right[opposite_angles].joints

opposite_arm.goto_state(des_pos_opposite_limb)
y.stop()
