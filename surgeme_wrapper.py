from surgeme_models import *
from debridement_models import *
from surgeme_splines import Surgeme_Splines
import time

class Surgemes():
    def __init__(self, robot='yumi', strategy='model'):
        self.strategy = strategy
        if self.strategy=='model' and robot=='yumi':
            self.execution = Surgeme_Models()
        elif self.strategy=='model' and robot=='taurus':
            self.execution = Debridement_Models()
        else:
            self.execution = Surgeme_Splines()
    
    ################# Add grippers and neutral and current pose
    def S1(self,final_pose,g_angle,limb):
        if robot=='yumi':
            self.execution.joint_orient(limb,g_angle)
            time.sleep(2)
            self.execution.surgeme1(1,final_pose,limb)
        print("Finished Approach")
        time.sleep(1)


    def S2(self,final_pose,limb):
        if robot=='yumi':
            self.execution.surgeme2(1,final_pose,limb)
        print("Finished Grasping")

    def S3(self,limb):
        if robot=='yumi':
            self.execution.surgeme3(1,limb)
            self.execution.ret_to_neutral(limb)
        print("Finished Lift")
        # if self.strategy == 'model':
        # self.execution.ret_to_neutral_angles(limb)

    def S4(self,limb):
        if robot=='yumi':
            self.execution.surgeme4(limb)
        print ("Finished Go TO Transfer ")

    def S5(self,limb,opposite_limb):
        if robot=='yumi':
            self.execution.surgeme5(limb)
            self.execution.ret_to_neutral(limb)
            self.execution.ret_to_neutral(opposite_limb)
        # self.execution.ret_to_neutral_angles(limb)
        # self.execution.ret_to_neutral_angles(opposite_limb)
        transfer_flag = 1
        print("Finsihed Transfer ")
        return transfer_flag

    def S6(self,drop_pt,limb):
        if robot=='yumi':
            self.execution.surgeme6(drop_pt,limb)
        print ("Finished Approach ")
        time.sleep(1.5)

    def S7(self,limb,opposite_limb, drop_ht):
        if robot=='yumi':
            time.sleep(1)
            self.execution.surgeme7(limb, drop_ht)
            # self.execution.ret_to_neutral_angles(opposite_limb)
            self.execution.ret_to_neutral(limb)
        print("Finished align and Drop")

    def get_curr_pose(self,limb):
        return self.execution.get_curr_pose(limb)

    def get_curr_joints(self,limb):
        return self.execution.get_curr_joints(limb)

    def ret_to_neutral_angles(self, limb):
        return self.execution.ret_to_neutral_angles(limb)

    def arm_open(self, limb):
        if limb == 'left':
            self.execution.left_open()
        else:
            self.execution.right_open()

    def arm_close(self, limb):
        if limb == 'left':
            self.execution.left_close()
        else:
            self.execution.right_close()
            
    def stop(self):
        self.execution.stow()
        time.sleep(10)
        self.execution.approach()
        time.sleep(10)
        self.execution.ready()
        time.sleep(10)

    def stow(self):
        if robot == "taurus":
            self.execution.stow()

    def approach(self):
        if robot == "taurus":
            self.execution.approach()

    def ready(self):
        if robot == "taurus":
            self.execution.ready()

