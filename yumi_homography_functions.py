import pickle
import numpy as np


def yumi_to_world(vector,yumi_arm):
	# vector (ndim,*/) in yumi coordinates m
	# output is (ndim */)
	#returns the coordinates in the world system 
	# each column is one data point
	with open("yumi_homography.pkl", 'rb') as fopen:
		saved_model = pickle.load(fopen)

	M_left=saved_model['M_left']
	M_right=saved_model['M_right']

	b=vector.shape[1]

	print(vector.shape)
	poses=np.concatenate((vector,np.ones((1,b))),axis=0)
	if yumi_arm=="left":
		output=np.dot(M_left,poses)
	elif yumi_arm=="right":
		output=np.dot(M_right,poses)
	return output[:-1,:]

def world_to_yumi(vector,yumi_arm):
	# vector (ndim,*/) in world coordinates m
	# output is (ndim */)
	#returns the coordinates in the yumi coordinate system
	# each column is one data point
	
	fopen = open("yumi_homography.pkl","rb")
	saved_model = pickle.load(fopen)

	M_left=saved_model['M_left']
	M_right=saved_model['M_right']

	vector = vector.reshape(-1,1)
	b=vector.shape[1]
	# print(vector.shape)
	poses=np.concatenate((vector,np.ones((1,b))),axis=0)

	if yumi_arm=="left":
		output=np.dot(np.linalg.inv(M_left),poses)
	elif yumi_arm=="right":
		output=np.dot(np.linalg.inv(M_right),poses)
	return output[:-1,:]
