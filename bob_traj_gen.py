import sys
sys.path.insert(0, './')

import os
import numpy as np
import pandas as pd
from Utils import eulerian_to_cartesian, cartesian_to_eulerian, rotationBetweenVectors, interpolate_quaternions, degrees_to_radian, radian_to_degrees, compute_orthodromic_distance, store_dict_as_csv,\
     quaternion_to_eulerian, angle_between_planes
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import copy
import argparse
from itertools import permutations
import math
from random import sample
plt.rcParams["font.family"] = "serif"


# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def plot_3d_trace(positions, user, video, ref_positions=None):
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), alpha=0.1, color="r")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='parametric curve')
    if ref_positions is not None:
        ax.plot(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], label='reference curve')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('User: %s, Video: %s' % (user, video))
    ax.view_init(10,20)
##    ax.legend()
    #plt.savefig('~/Desktop/UoE/TRACK/%s.pdf' % (video))
    plt.show()

f = 'alltrajectories.txt'
XYZ_read = np.zeros((100,3))
with open(f, 'rb') as file:
    lines = file.readlines()
    count = 0
    for line in lines:
        count +=1
        if count > 99:
            break;
        xyz = []
        for elem in line.split():
            xyz.append(elem)
        XYZ_read[count, :] = [xyz[0], xyz[1], xyz[2]]

##plot_3d_trace(XYZ_read[1:, :], 'ref','ref')    
nT = 1000
NumSteps = 100
SD = 0.01

# create ideal points (az,el)
A = [(7/8)*np.pi,np.pi/6]
B = [np.pi,np.pi/4]
C = [(9/8)*np.pi,np.pi/6]

# manually create 6 possible trajectories: ABCA, ACBA, BACB, BCAB, CABC, CBAC
RawTrajAzEl = np.zeros((6,8));  # 6 cases * 8 az/el parameters
RawTrajAzEl[0,:] = [A[0],A[1],B[0],B[1],C[0],C[1],A[0],A[1]]
RawTrajAzEl[1,:] = [A[0],A[1],C[0],C[1],B[0],B[1],A[0],A[1]]
RawTrajAzEl[2,:] = [B[0],B[1],A[0],A[1],C[0],C[1],B[0],B[1]]   
RawTrajAzEl[3,:] = [B[0],B[1],C[0],C[1],A[0],A[1],B[0],B[1]]
RawTrajAzEl[4,:] = [C[0],C[1],A[0],A[1],B[0],B[1],C[0],C[1]]
RawTrajAzEl[5,:] = [C[0],C[1],B[0],B[1],A[0],A[1],C[0],C[1]]

for traj in range(nT):
    # select one of the 6 trajectories
    TrajIndex = np.random.randint(low=0,high=6,size=1)[0]  # generate a random index between 1..6
    SelectedTraj = RawTrajAzEl[TrajIndex,:]  # select one of the trajectories
    PerturbedTraj = (SelectedTraj + SD*np.random.rand(1,8))[0] # add random noise to the point positions
    PerturbedTraj[6] = PerturbedTraj[0] # get back to starting point
    PerturbedTraj[7] = PerturbedTraj[1]
    # generate the trajectories
    XYZ = np.zeros((NumSteps,3))  # coordinatees of the points on the unit sphere
    partition = int(NumSteps/3)
    for step in range(NumSteps-1):
        if step <= partition:  # split trajectory into 3 parts
            az = (PerturbedTraj[2] - PerturbedTraj[0]) * (step / partition) + PerturbedTraj[0]
            el = (PerturbedTraj[3] - PerturbedTraj[1]) * (step / partition) + PerturbedTraj[1]
        elif step <= 2*partition:
            az = (PerturbedTraj[4] - PerturbedTraj[2]) * ((step-partition) / partition) + PerturbedTraj[2]
            el = (PerturbedTraj[5] - PerturbedTraj[3]) * ((step-partition) / partition) + PerturbedTraj[3]
        else:
            az = (PerturbedTraj[6] - PerturbedTraj[4]) * ((step-2*partition) / partition) + PerturbedTraj[4]
            el = (PerturbedTraj[7]- PerturbedTraj[5]) * ((step-2*partition) / partition) + PerturbedTraj[5]
        XYZ[step+1,0] = np.cos(el)*np.sin(az) 
        XYZ[step+1,1] = -np.cos(el)*np.cos(az)
        XYZ[step+1,2] = np.sin(el)
##    plot_3d_trace(XYZ[1:,:], 'ref','ref')
