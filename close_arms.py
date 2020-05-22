from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
from time import sleep

y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
y.left.close_gripper(force=2,wait_for_res=False)
y.right.close_gripper(force=2,wait_for_res=False)
sleep(2)
y.stop()
