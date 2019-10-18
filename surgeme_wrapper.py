from surgeme_models import *
from surgeme_splines import Surgeme_Splines
import time

class Surgemes():
    def __init__(self, robot='yumi', strategy='model'):
        if strategy=='model':
            self.execution = Surgeme_Models()
        else:
            self.execution = Surgeme_Splines()

    ################# Add grippers and neutral and current pose  
    def S1(self,calign,g_angle,limb):
        self.execution.joint_orient(limb,g_angle)
        self.execution.surgeme1(1,calign,limb)
        time.sleep(1)
        print("Finished Approach")
        time.sleep(2)

    def S2(self,align,limb):
        self.execution.surgeme2(1,align,limb)
        print("Finished Grasping")

    def S3(self,limb):
        self.execution.surgeme3(1,limb)
        print("FInished Lift")
        self.execution.ret_to_neutral_angles(limb)

    def S4(self,limb):
        self.execution.surgeme4(limb)
        print ("Finished Go TO Transfer ")

    def S5(self,limb,opposite_limb):
        self.execution.surgeme5(limb)
        print("Finsihed Transfer ")
        self.execution.ret_to_neutral_angles(limb)
        self.execution.ret_to_neutral_angles(opposite_limb)
        transfer_flag = 1
        return transfer_flag

    def S6(self,drop_pt,limb):
        self.execution.surgeme6(drop_pt,limb)
        print ("Finished Approach ")
        time.sleep(1.5)

    def S7(self,limb,opposite_limb):
        time.sleep(1)
        self.execution.surgeme7(limb)
        print("Finished align and Drop")
        self.execution.ret_to_neutral_angles(opposite_limb)

    def get_curr_pose(self,limb):
        return self.execution.get_curr_pose(limb)

    def ret_to_neutral_angles(self, limb):
        return self.execution.ret_to_neutral_angles(limb)

    def arm_open(self, limb):
        if limb == 'left':
            self.execution.left_open()
        else:
            self.execution.right_open()



