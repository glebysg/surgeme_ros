from surgeme_wrapper import Surgemes

limb = 'left'
robot="yumi"
opposite_limb = 'right' if limb == 'left' else 'left'
execution = Surgemes(strategy='model')
execution.ret_to_neutral_angles(opposite_limb)
execution.S4(limb)#Perform Go To transfer
# transfer_flag = execution.S5(limb, opposite_limb)

execution.stop()


