from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
import pickle as pkl
from time import sleep

y=YuMiRobot(include_right=True, log_state_histories=True, log_pose_histories=True)
y.left.move_gripper(0.005)
y.right.move_gripper(0.005)
sleep(2)
y.stop()
