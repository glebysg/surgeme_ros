from yumi_homography_functions import *
import numpy as np
from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState

#GUYS THIS IS THE EXAMPLE ABOUT HOW TO USE THE HOMOGRAPHY, THIS ONE SHOULD WORK!
# JUST IMPORT FUNCTIONS FROM SCRIPT yumi_homography_functions.py, that script loads the pickle file
# the pickle file is created by the script yumi_create_homography

def main():
	global y
	y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
	DELTARIGHT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0])
	DELTALEFT=RigidTransform(translation=[0, 0, 0.205], rotation=[1, 0, 0, 0]) #old version version is 0.32
	y.left.set_tool(DELTALEFT)
	y.right.set_tool(DELTARIGHT)
	y.set_v(40)
	y.set_z('z100')
	# close arms
	y.left.close_gripper(force=2,wait_for_res=False)
	y.right.close_gripper(force=2,wait_for_res=False)
	go_delta(np.array([[0,0,-0.00]]),"left")
	go_delta(np.array([[0,0.12,-0.00]]),"left")
	go_delta(np.array([[0.096,0,-0.00]]),"left")
	go_delta(np.array([[0.096,0.12,-0.00]]),"left")
	go_delta(np.array([[0.192,0,-0.00]]),"left")
	go_delta(np.array([[0.192,0.12,-0.00]]),"left")
	
	# go_delta(np.array([[0,0,-0.002]]),"right")
	# go_delta(np.array([[0,0.12,-0.002]]),"right")
	# go_delta(np.array([[0.096,0,-0.002]]),"right")
	# go_delta(np.array([[0.096,0.12,-0.002]]),"right")
	# go_delta(np.array([[0.192,0,-0.002]]),"right")
	# go_delta(np.array([[0.192,0.12,-0.002]]),"right")
	# go_delta(np.array([[0,0,-0.002]]),"right")
	#point 16?
	# go_delta(np.array([[0.048,0.072,-0.04208]]),"right")
	#new point
	# go_delta(np.array([[0,0,-0.002]]),"right")

def go_delta(point,arm):
	c=np.transpose(point)
	out=world_to_yumi(c,arm)
	pose=out.transpose()
	if arm=="left":
		pose_left=y.left.get_pose()
		delta=pose-pose_left.translation
		print("delta",delta)

		y.left.goto_pose_delta([delta[0,0],delta[0,1],delta[0,2]])

	if arm=="right":
		pose_right=y.right.get_pose()
		delta=pose-pose_right.translation
		print("delta",delta)

		y.right.goto_pose_delta([delta[0,0],delta[0,1],delta[0,2]])

if __name__ == '__main__':
    main()



