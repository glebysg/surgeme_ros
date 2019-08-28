import logging
import time
import os
import unittest
import numpy as np
import numpy
import copy
import sys
from autolab_core import RigidTransform
from yumipy import YuMiConstants as YMC
from yumipy import YuMiRobot, YuMiState
import IPython
import argparse
import pickle as pkl
import math

'''
    ALL DIMENSIONS ARE IN METERS
    THE FOLLOWING SCRIPT CERATED THE HOMOGRAPHY BETWEEN 12 POINTS IN THE WORLD COORDINATES MEASURED BY HAND
    AND 12 POINTS FROM EACH ROBOT ARM SAVED IN PICKLE FILES

    FIRST SAVE THE 12 POINTS FOR EACH YUMI ARM USING THE FILE  yumi_calibration_poses.py
'''


def main():
    global M_left
    global M_right

    world_points=np.array([[0,0,0],
            [0.08,0,0],
            [0.16,0,0],
            [0,0.10,0],
            [0.08,0.10,0],
            [0.16,0.10,0],
            [0,0,-0.0334],
            [0.08,0,-0.0334],
            [0.16,0,-0.0334],
            [0,0.10,-0.0334],
            [0.08,0.10,-0.0334],
            [0.16,0.10,-0.0334]])
    world_points=np.transpose(world_points)

    pose_left=[]
    pose_right=[]

    # Creates the vector of yumi poses for both arms using the saved pickle files
    # the pickle files are saved with the script yumi_calibration_poses.py

    for i in range(12):
        arm_pose_left = pkl.load(open("data/homography/poses_left_%s" % str(i+1), "rb" ) )
        arm_pose_right = pkl.load(open("data/homography/poses_right_%s" % str(i+1), "rb" ) )
        pose_left.append(arm_pose_left['left'].translation)
        pose_right.append(arm_pose_right["right"].translation)

    pose_left=np.array(pose_left)
    pose_right=np.array(pose_right)

    print pose_left[:,:3]

    # print pose_left
    pose_left=np.transpose(pose_left)
    pose_right=np.transpose(pose_right)

    # create transformation from yumi to world
    M_left=affine_matrix_from_points(pose_left, world_points, scale=False, usesvd=True)
    M_right=affine_matrix_from_points(pose_right, world_points, scale=False, usesvd=True)

    l_L1_difference_yumi_to_world=[]
    l_L1_difference_world_to_yumi=[]
    r_L1_difference_yumi_to_world=[]
    r_L1_difference_world_to_yumi=[]
    point=1
    output=yumi_to_world(pose_left,"left")
    output2=world_to_yumi(world_points,"left")
    output3=yumi_to_world(pose_right,"right")
    output4=world_to_yumi(world_points,"right")

    a,b=output.shape

    for i in range(b):
    	l_L1_difference_yumi_to_world.append(numpy.linalg.norm((output[:,i] - world_points[:,i]), ord=1))
    	l_L1_difference_world_to_yumi.append(numpy.linalg.norm((output2[:,i] - pose_left[:,i]), ord=1))
    	r_L1_difference_yumi_to_world.append(numpy.linalg.norm((output3[:,i] - world_points[:,i]), ord=1))
    	r_L1_difference_world_to_yumi.append(numpy.linalg.norm((output4[:,i] - pose_right[:,i]), ord=1))

    # print "LEFT L1 error yumi to world in mm",np.array(l_L1_difference_yumi_to_world)*1000
    print "LEFT MEAN L1 error yumi to world in mm ------->",np.mean(np.array(l_L1_difference_yumi_to_world)*1000,axis=0)
    # print "LEFT L1 error world to yumi mm",np.array(l_L1_difference_world_to_yumi)*1000
    print "LEFT MEAN L1 error world to yumi in mm ------>",np.mean(np.array(l_L1_difference_world_to_yumi)*1000,axis=0)

    # print "RIGHT L1 error yumi to world in mm",np.array(r_L1_difference_yumi_to_world)*1000
    print "RIGHT MEAN L1 error yumi to world in mm ------->",np.mean(np.array(r_L1_difference_yumi_to_world)*1000,axis=0)
    # print "RIGHT L1 error world to yumi mm",np.array(r_L1_difference_world_to_yumi)*1000
    print "RIGHT MEAN L1 error world to yumi in mm ------>",np.mean(np.array(r_L1_difference_world_to_yumi)*1000,axis=0)

    # print "estimated world coordinates LEFT",output[:,:]
    # print "real world coordinates LEFT",world_points[:,:]
    # print "estimated yumi coordinates LEFT", output2[:,:]
    # print "real yumi coordinates LEFT",pose_left[:,:]

    import pickle
    save_model = {'M_left':M_left,'M_right':M_right}
    pickle_out = open("yumi_homography.pkl","wb")
    pickle.dump(save_model, pickle_out)
    pickle_out.close()

def yumi_to_world(vector,yumi_arm):
    # vector (ndim,*/) in yumi coordinates m
    # output is (ndim */)
    #returns the coordinates in the world system 
    # each column is one data point

    b=vector.shape[1]
    global M_left
    global M_right
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

    b=vector.shape[1]
    global M_left
    global M_right
    print(vector.shape)
    poses=np.concatenate((vector,np.ones((1,b))),axis=0)

    if yumi_arm=="left":
        output=np.dot(np.linalg.inv(M_left),poses)
    elif yumi_arm=="right":
        output=np.dot(np.linalg.inv(M_right),poses)
    return output[:-1,:]

# from https://github.com/matthew-brett/transforms3d/blob/master/original/transformations.py 
def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.
    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.
    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.
    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.
    The returned matrix performs rotation, translation and uniform scaling
    (if specified).
    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[   0.14549,    0.00062,  675.50008],
           [   0.00048,    0.14094,   53.24971],
           [   0.     ,    0.     ,    1.     ]])
    >>> T = translation_matrix(numpy.random.random(3)-0.5)
    >>> R = random_rotation_matrix(numpy.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (numpy.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = numpy.dot(M, v0)
    >>> v0[:3] += numpy.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    More examples in superimposition_matrix()
    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=True)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -numpy.mean(v0, axis=1)
    M0 = numpy.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -numpy.mean(v1, axis=1)
    M1 = numpy.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = numpy.concatenate((v0, v1), axis=0)
        u, s, vh = numpy.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = numpy.dot(C, numpy.linalg.pinv(B))
        t = numpy.concatenate((t, numpy.zeros((ndims, 1))), axis=1)
        M = numpy.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = numpy.linalg.svd(numpy.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = numpy.dot(u, vh)
        if numpy.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= numpy.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = numpy.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = numpy.sum(v0 * v1, axis=1)
        xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = numpy.linalg.eigh(N)
        q = V[:, numpy.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))

    # move centroids back
    M = numpy.dot(numpy.linalg.inv(M1), numpy.dot(M, M0))
    M /= M[ndims, ndims]
    return M


if __name__ == '__main__':
    main()

