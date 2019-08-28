from yumi_homography_functions import *
import numpy as np

#GUYS THIS IS THE EXAMPLE ABOUT HOW TO USE THE HOMOGRAPHY, THIS ONE SHOULD WORK!
# JUST IMPORT FUNCTIONS FROM SCRIPT yumi_homography_functions.py, that script loads the pickle file
# the pickle file is created by the script yumi_create_homography

point=np.array([[0.08,0,0]])
c=np.transpose(point)

out=world_to_yumi(c,"left")
print(out)
out_2=yumi_to_world(out,"left")
print(out_2)



