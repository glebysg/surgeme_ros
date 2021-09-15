import pickle
import numpy as np


def taurus_to_world(vector,yumi_arm):
    # vector (ndim,*/) in yumi coordinates m
    # output is (ndim */)
    #returns the coordinates in the world system 
    # each column is one data point

    b=vector.shape[1]
    #global M_left
    global M_right
    print(vector.shape)
    poses=np.concatenate((vector,np.ones((1,b))),axis=0)
    if yumi_arm=="left":
        # output=np.dot(M_left,poses)
        pass
    elif yumi_arm=="right":
        output=np.dot(M_right,poses)
    return output[:-1,:]

def world_to_taurus(vector,arm='right'):
    # vector (ndim,*/) in world coordinates m
    # output is (ndim */)
    #returns the coordinates in the yumi coordinate system
    # each column is one data point
    fopen = open("taurus_homography.pkl","rb")
    saved_model = pickle.load(fopen)

    #M_left=saved_model['M_left']
    M_right=saved_model['M_right']

    vector = vector.reshape(-1,1)
    print(vector)

    b=vector.shape[1]

    print(vector.shape)

    poses=np.concatenate((vector,np.ones((1,b))),axis=0)

    if arm=="left":
        #output=np.dot(np.linalg.inv(M_left),poses)
        pass
    elif arm=="right":
        # print("Norm of inv", np.linalg.norm(M_right))
        output=np.dot(np.linalg.inv(M_right),poses)
    return output[:-1,:]
