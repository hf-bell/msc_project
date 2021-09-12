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
#import tensorflow_probability as tfp
import vg
from scipy.ndimage import gaussian_filter
plt.rcParams["font.family"] = "serif"

ROOT_FOLDER = './David_MMSys_18/dataset/'
OUTPUT_FOLDER = './David_MMSys_18/sampled_dataset'

OUTPUT_FOLDER_ORIGINAL_XYZ = './David_MMSys_18/original_dataset_xyz'

OUTPUT_TRUE_SALIENCY_FOLDER = './David_MMSys_18/true_saliency'

SAMPLING_RATE = 0.2

NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

VIDEOS = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight', '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino', '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']

# From "David_MMSys_18/dataset/Videos/Readme_Videos.md"
# Text files are provided with scanpaths from head movement with 100 samples per observer
NUM_SAMPLES_PER_USER = 100

def get_orientations_for_trace(filename):
    dataframe = pd.read_csv(filename, engine='python', header=0, sep=',')
    data = dataframe[[' longitude', ' latitude']]
    return data.values

def get_time_stamps_for_trace(filename):
    dataframe = pd.read_csv(filename, engine='python', header=0, sep=',')
    data = dataframe[' start timestamp']
    return data.values

# returns the frame rate of a video using openCV
# ToDo Copied (changed videoname to videoname+'_saliency' and video_path folder) from Xu_CVPR_18/Reading_Dataset (Author: Miguel Romero)
def get_frame_rate(videoname):
    video_mp4 = videoname+'_saliency.mp4'
    video_path = os.path.join(ROOT_FOLDER, 'content/saliency', video_mp4)
    video = cv2.VideoCapture(video_path)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

# Generate a dataset first with keys per user, then a key per video in the user and then for each sample a set of three keys
# 'sec' to store the time-stamp. 'yaw' to store the longitude, and 'pitch' to store the latitude
def get_original_dataset():
    dataset = {}
    for root, directories, files in os.walk(os.path.join(ROOT_FOLDER, 'Videos/H/Scanpaths')):
        for enum_trace, filename in enumerate(files):
            print('get head orientations from original dataset traces for video', enum_trace, '/', len(files))
            splitted_filename = filename.split('_')
            video = '_'.join(splitted_filename[:-1])
            file_path = os.path.join(root, filename)
            positions_all_users = get_orientations_for_trace(file_path)
            time_stamps_all_users = get_time_stamps_for_trace(file_path)
            num_users = int(positions_all_users.shape[0]/NUM_SAMPLES_PER_USER)
            for user_id in range(num_users):
                user = str(user_id)
                if user not in dataset.keys():
                    dataset[user] = {}
                positions = positions_all_users[user_id * NUM_SAMPLES_PER_USER:(user_id + 1) * (NUM_SAMPLES_PER_USER)]
                time_stamps = time_stamps_all_users[user_id * NUM_SAMPLES_PER_USER:(user_id + 1) * (NUM_SAMPLES_PER_USER)]
                samples = []
                for pos, t_stamp in zip(positions, time_stamps):
                    samples.append({'sec': t_stamp/1000.0, 'yaw': pos[0], 'pitch': pos[1]})
                dataset[user][video] = samples
    return dataset

# From "dataset/Videos/Readme_Videos.md"
# Latitude and longitude positions are normalized between 0 and 1 (so they should be multiplied according to the
# resolution of the desired equi-rectangular image output dimension).
# Participants started exploring omnidirectional contents either from an implicit longitudinal center
# (0-degrees and center of the equirectangular projection) or from the opposite longitude (180-degrees).
def transform_the_degrees_in_range(yaw, pitch):
    yaw = yaw*2*np.pi
    pitch = pitch*np.pi
    return yaw, pitch

# Performs the opposite transformation than transform_the_degrees_in_range
# Transform the yaw values from range [0, 2pi] to range [0, 1]
# Transform the pitch values from range [0, pi] to range [0, 1]
def transform_the_radians_to_original(yaw, pitch):
    yaw = yaw/(2*np.pi)
    pitch = pitch/np.pi
    return yaw, pitch


# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def recover_original_angles_from_quaternions_trace(quaternions_trace):
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        sample_new = quat_rot.rotate(orig_vec)
        restored_yaw, restored_pitch = cartesian_to_eulerian(sample_new[0], sample_new[1], sample_new[2])
        restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
        angles_per_video.append(np.array([restored_yaw, restored_pitch]))
    return np.array(angles_per_video)

def recover_original_angles_from_xyz_trace(xyz_trace):
    angles_per_video = []
    for sample in xyz_trace:
        restored_yaw, restored_pitch = cartesian_to_eulerian(sample[1], sample[2], sample[3])
        restored_yaw, restored_pitch = transform_the_radians_to_original(restored_yaw, restored_pitch)
        angles_per_video.append(np.array([restored_yaw, restored_pitch]))
    return np.array(angles_per_video)

# ToDo Copied exactly from Xu_PAMI_18/Reading_Dataset (Author: Miguel Romero)
def recover_xyz_from_quaternions_trace(quaternions_trace):
    angles_per_video = []
    orig_vec = np.array([1, 0, 0])
    for sample in quaternions_trace:
        quat_rot = Quaternion(sample[1:])
        sample_new = quat_rot.rotate(orig_vec)
        angles_per_video.append(sample_new)
    return np.concatenate((quaternions_trace[:, 0:1], np.array(angles_per_video)), axis=1)


# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
# Return the dataset
# yaw = 0, pitch = pi/2 is equal to (1, 0, 0) in cartesian coordinates
# yaw = pi/2, pitch = pi/2 is equal to (0, 1, 0) in cartesian coordinates
# yaw = pi, pitch = pi/2 is equal to (-1, 0, 0) in cartesian coordinates
# yaw = 3*pi/2, pitch = pi/2 is equal to (0, -1, 0) in cartesian coordinates
# yaw = Any, pitch = 0 is equal to (0, 0, 1) in cartesian coordinates
# yaw = Any, pitch = pi is equal to (0, 0, -1) in cartesian coordinates
def get_xyz_dataset(sampled_dataset):
    dataset = {}
    for user in sampled_dataset.keys():
        dataset[user] = {}
        for video in sampled_dataset[user].keys():
            dataset[user][video] = recover_xyz_from_quaternions_trace(sampled_dataset[user][video])
            print(dataset[user][video].shape)
    return dataset

# Store the dataset in xyz coordinates form into the folder_to_store
def store_dataset(xyz_dataset, folder_to_store):
    for user in xyz_dataset.keys():
        for video in xyz_dataset[user].keys():
            video_folder = os.path.join(folder_to_store, video)
            # Create the folder for the video if it doesn't exist
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            path = os.path.join(video_folder, user)
            df = pd.DataFrame(xyz_dataset[user][video])
            if np.isnan(df.values[:,1:]).any():
                print("raw",xyz_dataset[user][video][:,1:])
                print("data",np.isnan(df.values[:,1:]).any())
                plot_3d_trace(xyz_dataset[user][video][:,1:], video,user, df.values[:,1:])

            df.to_csv(path, header=False, index=False)


def compare_integrals(original_dataset, sampled_dataset):
    error_per_trace = []
    traces = []
    for user in original_dataset.keys():
        for video in original_dataset[user].keys():
            integ_yaws_orig = 0
            integ_pitchs_orig = 0
            for count, sample in enumerate(original_dataset[user][video]):
                if count == 0:
                    prev_sample = original_dataset[user][video][0]
                else:
                    dt = sample['sec'] - prev_sample['sec']
                    integ_yaws_orig += sample['yaw'] * dt
                    integ_pitchs_orig += sample['pitch'] * dt
                    prev_sample = sample
            angles_per_video = recover_original_angles_from_quaternions_trace(sampled_dataset[user][video])
            integ_yaws_sampl = 0
            integ_pitchs_sampl = 0
            for count, sample in enumerate(angles_per_video):
                if count == 0:
                    prev_time = sampled_dataset[user][video][count, 0]
                else:
                    dt = sampled_dataset[user][video][count, 0] - prev_time
                    integ_yaws_sampl += angles_per_video[count, 0] * dt
                    integ_pitchs_sampl += angles_per_video[count, 1] * dt
                    prev_time = sampled_dataset[user][video][count, 0]
            error_per_trace.append(np.sqrt(np.power(integ_yaws_orig-integ_yaws_sampl, 2) + np.power(integ_pitchs_orig-integ_pitchs_sampl, 2)))
            traces.append({'user': user, 'video': video})
    return error_per_trace, traces

### Check if the quaternions are good
def compare_sample_vs_original(original_dataset, sampled_dataset):
    for user in original_dataset.keys():
        for video in original_dataset[user].keys():
            pitchs = []
            yaws = []
            times = []
            for sample in original_dataset[user][video]:
                times.append(sample['sec'])
                yaws.append(sample['yaw'])
                pitchs.append(sample['pitch'])
            angles_per_video = recover_original_angles_from_xyz_trace(sampled_dataset[user][video])
            plt.subplot(1, 2, 1)
            plt.plot(times, yaws, label='yaw')
            plt.plot(times, pitchs, label='pitch')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(sampled_dataset[user][video][:, 0], angles_per_video[:, 0], label='yaw')
            plt.plot(sampled_dataset[user][video][:, 0], angles_per_video[:, 1], label='pitch')
            plt.legend()
            plt.show()

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
# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
# def plot_all_traces_in_3d(xyz_dataset, error_per_trace, traces):
#     indices = np.argsort(-np.array(error_per_trace))
#     for trace_id in indices:
#         trace = traces[trace_id]
#         user = trace['user']
#         video = trace['video']
#         plot_3d_trace(xyz_dataset[user][video][:, 1:])

def plot_all_traces_in_3d(xyz_dataset):
    for video in xyz_dataset.keys():
        for user in xyz_dataset[video].keys():
            plot_3d_trace(xyz_dataset[video][user][:, 1:], user, video)
    plt.show()

def slerp(p1,p2,t):
    omega = np.arccos( p1.dot(p2) )
##    if np.degrees(omega) > 30:
##        omega = 2*np.pi - omega
    sin_omega = np.sin(omega)
    t = t[:, np.newaxis]
    return ( np.sin( (1-t)*omega )*p1 + np.sin( t*omega )*p2 )/sin_omega


# ToDo Copied exactly from AVTrack360/Reading_Dataset (Author: Miguel Romero)
def create_sampled_dataset(original_dataset, rate):
    dataset = {}
    for enum_user, user in enumerate(original_dataset.keys()):
        dataset[user] = {}
        for enum_video, video in enumerate(original_dataset[user].keys()):
            print('creating sampled dataset', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
            sample_orig = np.array([1, 0, 0])
            data_per_video = []
            for sample in original_dataset[user][video]:
                sample_yaw, sample_pitch = transform_the_degrees_in_range(sample['yaw'], sample['pitch'])
                sample_new = eulerian_to_cartesian(sample_yaw, sample_pitch)
                quat_rot = rotationBetweenVectors(sample_orig, sample_new)
                # append the quaternion to the list
                data_per_video.append([sample['sec'], quat_rot[0], quat_rot[1], quat_rot[2], quat_rot[3]])
                # update the values of time and sample
            # interpolate the quaternions to have a rate of 0.2 secs
            data_per_video = np.array(data_per_video)
            dataset[user][video] = interpolate_quaternions(data_per_video[:, 0], data_per_video[:, 1:], rate=rate)
            print(dataset[user][video].shape)
    return dataset

def sample_trig_loop(npoints):
    theta = np.random.uniform(-2*np.pi,2*np.pi,npoints)
    phi = np.random.uniform(-np.pi,np.pi,npoints)
##    phi = np.repeat(np.pi/2, npoints)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.array([x,y,z])


def create_simple_synthetic_training_set(xyz_dataset):
    user = '4'
    video = '1_PortoRiverside'
    orig_trace = copy.deepcopy(xyz_dataset[user][video])
    nT = 500
    NumSteps = 101
    SD = 0.05
    synthetic_train = {}

    # create ideal points (az,el)
##    A = [(7/8)*np.pi,np.pi/6]
    A = [(5/8)*np.pi,np.pi/6]
    B = [np.pi,np.pi/4]
    C = [(9/8)*np.pi,np.pi/6]

##    A = [np.pi,np.pi/6]
##    B = [np.pi,np.pi/12]
##    C = [(3/8)*np.pi,np.pi/12]

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
        PerturbedTraj = (SelectedTraj + SD*np.random.uniform(size=(1,8)))[0] # add random noise to the point positions
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

        trace = copy.deepcopy(orig_trace)
        trace[:,1:] = XYZ[1:, :]
        name = 'traj_%s'%(traj)
        video = 'simple_%s_1_change'%SD
        synthetic_train[name] = {}
        synthetic_train[name][video] = trace
        print("\n \n \n ----------- %s LOGGED ----------------\n \n \n"%(name))
##        plot_3d_trace(XYZ[1:,:], name, video)

##    plot_all_traces_in_3d(synthetic_train)
    SYNTHETIC_OUTPUT_FOLDER = './David_MMSys_18/simple_training_set'
    store_dataset(synthetic_train,SYNTHETIC_OUTPUT_FOLDER)
            
def create_synthetic_training_set(xyz_dataset):
    user = '4'
    video = '1_PortoRiverside'
##    plot_3d_trace(xyz_dataset[user][video][:,1:], user, video)
##    np.random.seed(0)
    users = ['50', '37', '24']
    videos = ['4_Ocean', '2_Diner']
    user = sample(users,1)[0]
    video = sample(videos, 1)[0]
    print("User: %s, Video: %s" %(user,video))
    trace_shape = np.zeros((len(xyz_dataset[user][video][:,1:]),len(xyz_dataset[user][video][0,1:])))
    orig_trace = copy.deepcopy(xyz_dataset[user][video])
##    plot_3d_trace(
##            orig_trace[:,1:],
##            user,
##            video,
##            )
    synthetic_train = {}
    for train_eg in range(1000):
        positions = sample_trig_loop(2000)
        init_constraint = False
        equat_constraint = False
        rot_constraint = False
        used_init_idxs = []
        traj = []
        while not init_constraint:
            position_idx = np.random.randint(low=0,high=1000,size=1)[0]
            if position_idx in used_init_idxs:
                print("used")
                continue;
            else:
                used_init_idxs.append(position_idx)
            init_pos = copy.deepcopy(positions[:,position_idx])
            init_theta, init_phi = cartesian_to_eulerian(init_pos[0], init_pos[1], init_pos[2])
            if abs(init_phi) > np.pi/2 - np.pi/8 and abs(init_phi) < np.pi/2 + np.pi/8:
                init_constraint = True
        traj.append(init_pos)

        for k in range(4):
            print("point", k)
            rot_constraint = False
            equat_constraint = False
            used_init_idxs = []
            count = 0
            resample = 0
            while (not equat_constraint) or (not rot_constraint):
                count += 1
                if count > 500:
                    count=0
                    resample += 1
                    print("RESAMpLE------------------------------------------------")
                    positions = sample_trig_loop(2000)
                position_idx = np.random.randint(low=0,high=2000,size=1)[0]
                if position_idx in used_init_idxs:
                    print("used")
                    continue;
                next_pos = copy.deepcopy(positions[:,position_idx])
                # constrain distance from equator
                if any(np.array_equal(next_pos,pos) for pos in traj):
                    print("same as previous position")
                    continue;
                    print("past continue")
                next_theta, next_phi = cartesian_to_eulerian(next_pos[0], next_pos[1], next_pos[2])
##                if abs(next_pos)[2] < 0.125:
##                    equat_constraint = True
                if abs(next_phi) > np.pi/2 - np.pi/8 and abs(next_phi) < np.pi/2 + np.pi/8:
                       equat_constraint = True
                else:
                    continue;
                if len(traj) == 1:
                    q = rotationBetweenVectors(init_pos, next_pos)
                    q_rot = R.from_quat([q[0],q[1],q[2],q[3]])
                    q_euler = q_rot.as_euler('xyz',degrees=True)
                    if any(abs(angle) > 40 for angle in q_euler[1:]):
                        vec_ab = np.array([next_pos[0] - init_pos[0],
                                       next_pos[1] - init_pos[1],
                                       next_pos[2] - init_pos[2]])
                        vec_bc = np.array([next_pos[0] - traj[k][0],
                                       next_pos[1] - traj[k][1],
                                       next_pos[2] - traj[k][2]])
                        v1mag = np.sqrt(vec_ab[0]**2 + vec_ab[1]**2 + vec_ab[2]**2)
                        v1norm = np.array([vec_ab[0] / v1mag,
                                           vec_ab[1] / v1mag,
                                           vec_ab[2] / v1mag])

                        v2mag = np.sqrt(vec_bc[0]**2 + vec_bc[1]**2 + vec_bc[2]**2)
                        v2norm = np.array([vec_bc[0] / v2mag,
                                           vec_bc[1] / v2mag,
                                           vec_bc[2] / v2mag])

                        res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
                        angle = np.arccos(res)
                        traj.append(copy.deepcopy(next_pos))
                        rot_constraint = True
                        break;
                    else:
                        continue;
                else:
                    angle = angle_between_planes(traj[k-1], traj[k], next_pos)
                    
                    #print(np.degrees(angle))
                    if radian_to_degrees(angle) > 50 and radian_to_degrees(angle) < 60:
                        print("passed angle",np.degrees(angle))
                        traj.append(copy.deepcopy(next_pos))
                        rot_constraint = True
                        break;
                    else:
                        continue;
        trace = copy.deepcopy(orig_trace)

        _interp_a = traj[0]
        _interp_b = traj[1]
        _interp_c = traj[2]
        _interp_d = traj[3]
        _interp_e = traj[4]  
        t = np.linspace(0,1,25)
        t_half = np.linspace(0,1,50)
        traj_1 = slerp(_interp_a, _interp_b, t)
        traj_2 = slerp(_interp_b, _interp_c, t)
        traj_3 = slerp(_interp_c, _interp_d, t)
        traj_4 = slerp(_interp_d, _interp_e, t)
        traj_ac= slerp(_interp_a, _interp_c,t_half)
        traj_bd= slerp(_interp_b, _interp_d,t_half)
        traj_ce= slerp(_interp_c, _interp_e,t_half)
        traj = np.concatenate((traj_1, traj_2, traj_3,traj_4),axis=0)
    ##    traj[:50, :] = np.mean((traj[:50,:], traj_ac),axis=0)
    ##    traj[25:75, :] = np.mean((traj[25:75,:], traj_bd),axis=0)
    ##    traj[50:, :] = np.mean((traj[:50,:], traj_ce),axis=0)
    ##    from scipy import signal
        import bottleneck as bn
        for axis in range(len(traj.shape)):
            traj[20:30,axis] = bn.move_mean(traj[20:30,axis], window=5, min_count=1)
            traj[45:55,axis] = bn.move_mean(traj[45:55,axis], window=5, min_count=1)
            traj[70:80,axis] = bn.move_mean(traj[70:80,axis], window=5, min_count=1)
            
        trace[:,1:] = traj
        name = 'traj_%s'%(train_eg)
        video = 'HSTD_training'
        synthetic_train[name] = {}
        synthetic_train[name][video] = trace
        print("\n \n \n ----------- %s LOGGED ----------------\n \n \n"%(name))
##        plot_3d_trace(
##            trace[:,1:],
##            name,
##            video,
##            )

    SYNTHETIC_OUTPUT_FOLDER = './David_MMSys_18/HSTD_training_set'
    store_dataset(synthetic_train,SYNTHETIC_OUTPUT_FOLDER)


def create_synthetic_test_set(xyz_dataset):
##    np.random.seed(0)
    users = ['50', '37', '24']
    videos = ['4_Ocean', '2_Diner']
    user = sample(users,1)[0]
    video = sample(videos, 1)[0]
    print("User: %s, Video: %s" %(user,video))
    trace_shape = np.zeros((len(xyz_dataset[user][video][:,1:]),len(xyz_dataset[user][video][0,1:])))
    orig_trace = copy.deepcopy(xyz_dataset[user][video])
    video = 'gt0_lt30'
    synthetic_test = {}
    for train_eg in range(200):
        positions = sample_trig_loop(2000)
        init_constraint = False
        equat_constraint = False
        rot_constraint = False
        used_init_idxs = []
        traj = []
        while not init_constraint:
            position_idx = np.random.randint(low=0,high=1000,size=1)[0]
            if position_idx in used_init_idxs:
                print("used")
                continue;
            else:
                used_init_idxs.append(position_idx)
            init_pos = copy.deepcopy(positions[:,position_idx])
##            if abs(init_pos[2]) < 0.125:
##                    init_constraint = True
            init_theta, init_phi = cartesian_to_eulerian(init_pos[0], init_pos[1], init_pos[2])
            if abs(init_phi) > np.pi/2 - np.pi/8 and abs(init_phi) < np.pi/2 + np.pi/8:
                init_constraint = True
        traj.append(init_pos)

        for k in range(4):
            print("point", k)
            rot_constraint = False
            equat_constraint = False
            used_init_idxs = []
            count = 0
            resample = 0
            while (not equat_constraint) or (not rot_constraint):
                count += 1
                if count > 500:
                    count=0
                    resample += 1
                    print("RESAMpLE------------------------------------------------")
                    positions = sample_trig_loop(2000)
                position_idx = np.random.randint(low=0,high=2000,size=1)[0]
                if position_idx in used_init_idxs:
                    print("used")
                    continue;
                next_pos = copy.deepcopy(positions[:,position_idx])
                # constrain distance from equator
                if any(np.array_equal(next_pos,pos) for pos in traj):
                    print("same as previous position")
                    continue;
                    print("past continue")
                next_theta, next_phi = cartesian_to_eulerian(next_pos[0], next_pos[1], next_pos[2])
##                if abs(next_pos)[2] < 0.125:
##                    equat_constraint = True
                if abs(next_phi) > np.pi/2 - np.pi/8 and abs(next_phi) < np.pi/2 + np.pi/8:
                       equat_constraint = True
                else:
                    continue;

                # constrain angle from prev point
##                if len(traj) == 1:
##                   q = rotationBetweenVectors(init_pos, next_pos)
##                else:
##                    q = rotationBetweenVectors(traj[k], next_pos)
##                q_rot = R.from_quat([q[0],q[1],q[2],q[3]])
##                q_euler = q_rot.as_euler('xyz',degrees=True)
##                if any(abs(angle) > 40 for angle in q_euler[1:]):# \
##                   #or abs(q_euler[0]) > 91:
####                    print("not first angle constraint")
##                    continue;
##                else:
                    #print("satisfied first constraint")
                if len(traj) == 1:
                    q = rotationBetweenVectors(init_pos, next_pos)
                    q_rot = R.from_quat([q[0],q[1],q[2],q[3]])
                    q_euler = q_rot.as_euler('xyz',degrees=True)
                    if any(abs(angle) > 40 for angle in q_euler[1:]):
                        vec_ab = np.array([next_pos[0] - init_pos[0],
                                       next_pos[1] - init_pos[1],
                                       next_pos[2] - init_pos[2]])
                        vec_bc = np.array([next_pos[0] - traj[k][0],
                                       next_pos[1] - traj[k][1],
                                       next_pos[2] - traj[k][2]])
                        v1mag = np.sqrt(vec_ab[0]**2 + vec_ab[1]**2 + vec_ab[2]**2)
                        v1norm = np.array([vec_ab[0] / v1mag,
                                           vec_ab[1] / v1mag,
                                           vec_ab[2] / v1mag])

                        v2mag = np.sqrt(vec_bc[0]**2 + vec_bc[1]**2 + vec_bc[2]**2)
                        v2norm = np.array([vec_bc[0] / v2mag,
                                           vec_bc[1] / v2mag,
                                           vec_bc[2] / v2mag])

                        res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
                        angle = np.arccos(res)
                        traj.append(copy.deepcopy(next_pos))
                        rot_constraint = True
                        break;
                    else:
                        continue;
                else:
                    angle = angle_between_planes(traj[k-1], traj[k], next_pos)

                    #print(np.degrees(angle))
                    if np.degrees(angle) > 0 and np.degrees(angle) < 30:
                        print("passed angle",np.degrees(angle))
                        traj.append(copy.deepcopy(next_pos))
                        rot_constraint = True
                        break;
                    else:
                        continue;
        trace = copy.deepcopy(orig_trace)
    ##    sample_orig = np.array([0,0,1])
    ##    rate=0.2
    ##    traj_quats = [rotationBetweenVectors(sample_orig, pos) for pos in traj]
    ##    traj_quats = np.array([[quat[0],quat[1],quat[2],quat[3]] for quat in traj_quats])
    ##    print(traj_quats)
    ##    key_rots = R.from_quat(traj_quats)
    ##    orig_times = [0,5,10,15,19.8]
    ##    slerp = Slerp(orig_times, key_rots)
    ##    times = np.arange(orig_times[0], orig_times[-1]+rate/2.0, rate)
    ##    times[-1] = min(orig_times[-1], times[-1])
    ##    print(slerp(times))

##        # 10 point interp
##        _interp_a = traj[0]
##        _interp_b = traj[1]
##        _interp_c = traj[2]
##        _interp_d = traj[3]
##        _interp_e = traj[4]
##        _interp_f = traj[5]
##        _interp_g = traj[6]
##        _interp_h = traj[7]
##        _interp_i = traj[8]
##        _interp_j = traj[9]
##        _interp_k = traj[10]
##        t = np.linspace(0,1,10)
##        t_half = np.linspace(0,1,50)
##        traj_1 = slerp(_interp_a, _interp_b, t)
##        traj_2 = slerp(_interp_b, _interp_c, t)
##        traj_3 = slerp(_interp_c, _interp_d, t)
##        traj_4 = slerp(_interp_d, _interp_e, t)
##        traj_5 = slerp(_interp_e, _interp_f, t)
##        traj_6 = slerp(_interp_f, _interp_g, t)
##        traj_7 = slerp(_interp_g, _interp_h, t)
##        traj_8 = slerp(_interp_h, _interp_i, t)
##        traj_9 = slerp(_interp_i, _interp_j, t)
##        traj_10 = slerp(_interp_j, _interp_k, t) 
##        traj = np.concatenate((traj_1, traj_2, traj_3,traj_4,traj_5, traj_6,traj_7,traj_8,traj_9,traj_10),axis=0)

        # training set interp
        _interp_a = traj[0]
        _interp_b = traj[1]
        _interp_c = traj[2]
        _interp_d = traj[3]
        _interp_e = traj[4]  
        t = np.linspace(0,1,25)
        t_half = np.linspace(0,1,50)
        traj_1 = slerp(_interp_a, _interp_b, t)
        traj_2 = slerp(_interp_b, _interp_c, t)
        traj_3 = slerp(_interp_c, _interp_d, t)
        traj_4 = slerp(_interp_d, _interp_e, t)
        traj = np.concatenate((traj_1, traj_2, traj_3,traj_4),axis=0)

# 5 point interp
##        _interp_a = traj[0]
##        _interp_b = traj[1]
##        _interp_c = traj[2]
##        _interp_d = traj[3]
##        _interp_e = traj[4]
##        _interp_f = traj[5]
##        t = np.linspace(0,1,20)
##        t_half = np.linspace(0,1,50)
##        traj_1 = slerp(_interp_a, _interp_b, t)
##        traj_2 = slerp(_interp_b, _interp_c, t)
##        traj_3 = slerp(_interp_c, _interp_d, t)
##        traj_4 = slerp(_interp_d, _interp_e, t)
##        traj_5 = slerp(_interp_e, _interp_f, t)
##        traj = np.concatenate((traj_1, traj_2, traj_3,traj_4,traj_5),axis=0)

# 2 point interp
        _interp_a = traj[0]
        _interp_b = traj[1]
        _interp_c = traj[2]
        t = np.linspace(0,1,50)
        t_half = np.linspace(0,1,50)
        traj_1 = slerp(_interp_a, _interp_b, t)
        traj_2 = slerp(_interp_b, _interp_c, t)
        traj = np.concatenate((traj_1, traj_2),axis=0)
        
        import bottleneck as bn
        for axis in range(len(traj.shape)):
            traj[20:30,axis] = bn.move_mean(traj[20:30,axis], window=5, min_count=1)
            traj[45:55,axis] = bn.move_mean(traj[45:55,axis], window=5, min_count=1)
            traj[70:80,axis] = bn.move_mean(traj[70:80,axis], window=5, min_count=1)
            
            # 10 pos smoothing
##            traj[5:15,axis] = bn.move_mean(traj[5:15,axis], window=5, min_count=1)
##            traj[15:25,axis] = bn.move_mean(traj[15:25,axis], window=5, min_count=1)
##            traj[25:35,axis] = bn.move_mean(traj[25:35,axis], window=5, min_count=1)
##            traj[35:45,axis] = bn.move_mean(traj[35:45,axis], window=5, min_count=1)
##            traj[45:55,axis] = bn.move_mean(traj[45:55,axis], window=5, min_count=1)
##            traj[55:65,axis] = bn.move_mean(traj[55:65,axis], window=5, min_count=1)
##            traj[65:75,axis] = bn.move_mean(traj[65:75,axis], window=5, min_count=1)
##            traj[75:85,axis] = bn.move_mean(traj[75:85,axis], window=5, min_count=1)
##            traj[85:95,axis] = bn.move_mean(traj[85:95,axis], window=5, min_count=1)

            # 2 pos smoothing
##            traj[45:55,axis] = bn.move_mean(traj[45:55,axis], window=5, min_count=1)

##            # 5 pos smoothing
##            traj[15:25,axis] = bn.move_mean(traj[15:25,axis], window=5, min_count=1)
##            traj[35:45,axis] = bn.move_mean(traj[35:45,axis], window=5, min_count=1)
##            traj[55:65,axis] = bn.move_mean(traj[55:65,axis], window=5, min_count=1)
##            traj[75:85,axis] = bn.move_mean(traj[75:85,axis], window=5, min_count=1)

        print("HELLO")
        trace[:,1:] = traj
        name = 'traj_%s'%(train_eg)
        plot_3d_trace(
            trace[:,1:],
            name,
            video,
            )
        synthetic_test[name] = {}
        synthetic_test[name][video] = trace
        print("\n \n \n ----------- %s LOGGED ----------------\n \n \n"%(name))

    
    SYNTHETIC_OUTPUT_FOLDER = './David_MMSys_18/gt0_lt30_50_60'
    store_dataset(synthetic_test,SYNTHETIC_OUTPUT_FOLDER)
        
# ToDo, transform in a class this is the main function of this file
def create_and_store_sampled_dataset(plot_comparison=False, plot_3d_traces=False):
    original_dataset = get_original_dataset()
    sampled_dataset = create_sampled_dataset(original_dataset, rate=SAMPLING_RATE)
    if plot_comparison:
        compare_sample_vs_original(original_dataset, sampled_dataset)
    xyz_dataset = get_xyz_dataset(sampled_dataset)
    if plot_3d_traces:
        plot_all_traces_in_3d(xyz_dataset)
    store_dataset(xyz_dataset, OUTPUT_FOLDER)

def create_and_store_true_saliency(sampled_dataset):
    if not os.path.exists(OUTPUT_TRUE_SALIENCY_FOLDER):
        os.makedirs(OUTPUT_TRUE_SALIENCY_FOLDER)

    # Returns an array of size (NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL) with values between 0 and 1 specifying the probability that a tile is watched by the user
    # We built this function to ensure the model and the groundtruth tile-probabilities are built with the same (or similar) function
    def from_position_to_tile_probability_cartesian(pos):
        yaw_grid, pitch_grid = np.meshgrid(np.linspace(0, 1, NUM_TILES_WIDTH_TRUE_SAL, endpoint=False),
                                           np.linspace(0, 1, NUM_TILES_HEIGHT_TRUE_SAL, endpoint=False))
        yaw_grid += 1.0 / (2.0 * NUM_TILES_WIDTH_TRUE_SAL)
        pitch_grid += 1.0 / (2.0 * NUM_TILES_HEIGHT_TRUE_SAL)
        yaw_grid = yaw_grid * 2 * np.pi
        pitch_grid = pitch_grid * np.pi
        x_grid, y_grid, z_grid = eulerian_to_cartesian(theta=yaw_grid, phi=pitch_grid)
        great_circle_distance = np.arccos(np.maximum(np.minimum(x_grid * pos[0] + y_grid * pos[1] + z_grid * pos[2], 1.0), -1.0))
        gaussian_orth = np.exp((-1.0 / (2.0 * np.square(0.1))) * np.square(great_circle_distance))
        return gaussian_orth

    for enum_video, video in enumerate(VIDEOS):
        print('creating true saliency for video', video, '-', enum_video, '/', len(VIDEOS))
        real_saliency_for_video = []
        for x_i in range(NUM_SAMPLES_PER_USER):
            tileprobs_for_video_cartesian = []
            for user in sampled_dataset.keys():
                tileprobs_cartesian = from_position_to_tile_probability_cartesian(sampled_dataset[user][video][x_i, 1:])
                tileprobs_for_video_cartesian.append(tileprobs_cartesian)
            tileprobs_for_video_cartesian = np.array(tileprobs_for_video_cartesian)
            real_saliency_cartesian = np.sum(tileprobs_for_video_cartesian, axis=0) / tileprobs_for_video_cartesian.shape[0]
            real_saliency_for_video.append(real_saliency_cartesian)
        real_saliency_for_video = np.array(real_saliency_for_video)

        true_sal_out_file = os.path.join(OUTPUT_TRUE_SALIENCY_FOLDER, video)
        np.save(true_sal_out_file, real_saliency_for_video)

def load_sampled_dataset():
    list_of_videos = [o for o in os.listdir(OUTPUT_FOLDER) if not o.endswith('.gitkeep')]
    dataset = {}
    for video in list_of_videos:
        for user in [o for o in os.listdir(os.path.join(OUTPUT_FOLDER, video)) if not o.endswith('.gitkeep')]:
            if user not in dataset.keys():
                dataset[user] = {}
            path = os.path.join(OUTPUT_FOLDER, video, user)
            data = pd.read_csv(path, header=None)
            dataset[user][video] = data.values
    return dataset

def load_synthetic_dataset():
    OUTPUT_FOLDER = './David_MMSys_18/synthetic_dataset'
    list_of_videos = [o for o in os.listdir(OUTPUT_FOLDER) if not o.endswith('.gitkeep')]
    dataset = {}
    for video in list_of_videos:
        for user in [o for o in os.listdir(os.path.join(OUTPUT_FOLDER, video)) if not o.endswith('.gitkeep')]:
            if user not in dataset.keys():
                dataset[user] = {}
            path = os.path.join(OUTPUT_FOLDER, video, user)
            data = pd.read_csv(path, header=None)
            dataset[user][video] = data.values
    return dataset

def get_most_salient_points_per_video():
    from skimage.feature import peak_local_max
    most_salient_points_per_video = {}
    for video in VIDEOS:
        saliencies_for_video_file = os.path.join(OUTPUT_TRUE_SALIENCY_FOLDER, video+'.npy')
        saliencies_for_video = np.load(saliencies_for_video_file)
        most_salient_points_in_video = []
        for id, sal in enumerate(saliencies_for_video):
            coordinates = peak_local_max(sal, exclude_border=False, num_peaks=5)
            coordinates_normalized = coordinates / np.array([NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL])
            coordinates_radians = coordinates_normalized * np.array([np.pi, 2.0*np.pi])
            cartesian_pts = np.array([eulerian_to_cartesian(sample[1], sample[0]) for sample in coordinates_radians])
            most_salient_points_in_video.append(cartesian_pts)
        most_salient_points_per_video[video] = np.array(most_salient_points_in_video)
    return  most_salient_points_per_video

def predict_most_salient_point(most_salient_points, current_point):
    pred_window_predicted_closest_sal_point = []
    for id, most_salient_points_per_fut_frame in enumerate(most_salient_points):
        distances = np.array([compute_orthodromic_distance(current_point, most_sal_pt) for most_sal_pt in most_salient_points_per_fut_frame])
        closest_sal_point = np.argmin(distances)
        predicted_closest_sal_point = most_salient_points_per_fut_frame[closest_sal_point]
        pred_window_predicted_closest_sal_point.append(predicted_closest_sal_point)
    return pred_window_predicted_closest_sal_point

def most_salient_point_baseline(dataset):
    most_salient_points_per_video = get_most_salient_points_per_video()
    error_per_time_step = {}
    for enum_user, user in enumerate(dataset.keys()):
        for enum_video, video in enumerate(dataset[user].keys()):
            print('computing error for user', enum_user, '/', len(dataset.keys()), 'video', enum_video, '/', len(dataset[user].keys()))
            trace = dataset[user][video]
            for x_i in range(5, 75):
                model_prediction = predict_most_salient_point(most_salient_points_per_video[video][x_i+1:x_i+25+1], trace[x_i, 1:])
                for t in range(25):
                    if t not in error_per_time_step.keys():
                        error_per_time_step[t] = []
                    error_per_time_step[t].append(compute_orthodromic_distance(trace[x_i+t+1, 1:], model_prediction[t]))
    for t in range(25):
        print(t*0.2, np.mean(error_per_time_step[t]))

def create_original_dataset_xyz(original_dataset):
    dataset = {}
    for enum_user, user in enumerate(original_dataset.keys()):
        dataset[user] = {}
        for enum_video, video in enumerate(original_dataset[user].keys()):
            print('creating original dataset in format for', 'user', enum_user, '/', len(original_dataset.keys()), 'video', enum_video, '/', len(original_dataset[user].keys()))
            data_per_video = []
            for sample in original_dataset[user][video]:
                sample_yaw, sample_pitch = transform_the_degrees_in_range(sample['yaw'], sample['pitch'])
                sample_new = eulerian_to_cartesian(sample_yaw, sample_pitch)
                data_per_video.append([sample['sec'], sample_new[0], sample_new[1], sample_new[2]])
            print(np.array(data_per_video).shape)
            print(np.array(data_per_video)[:,:1])
            dataset[user][video] = np.array(data_per_video)
    return dataset

def create_and_store_original_dataset():
    original_dataset = get_original_dataset()
    original_dataset_xyz = create_original_dataset_xyz(original_dataset)
    store_dataset(original_dataset_xyz, OUTPUT_FOLDER_ORIGINAL_XYZ)

def get_traces_for_train_and_test():
    videos = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight',
              '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino',
              '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']

    # Fixing random state for reproducibility
    np.random.seed(7)

    videos_ids = np.arange(len(videos))
    users = np.arange(57)

    # Select at random the users for each set
    np.random.shuffle(users)
    num_train_users = int(len(users) * 0.5)
    users_train = users[:num_train_users]
    users_test = users[num_train_users:]

    videos_ids_train = [1, 3, 5, 7, 8, 9, 11, 14, 16, 18]
    videos_ids_test = [0, 2, 4, 13, 15]

    train_traces = []
    for video_id in videos_ids_train:
        for user_id in users_train:
            train_traces.append({'video': videos[video_id], 'user': str(user_id)})

    test_traces = []
    for video_id in videos_ids_test:
        for user_id in users_test:
            test_traces.append({'video': videos[video_id], 'user': str(user_id)})

    return train_traces, test_traces

def split_traces_and_store():
    train_traces, test_traces = get_traces_for_train_and_test()
    store_dict_as_csv('train_set', ['user', 'video'], train_traces)
    store_dict_as_csv('test_set', ['user', 'video'], test_traces)

if __name__ == "__main__":
    #print('use this file to create sampled dataset or to create true_saliency or to create original dataset in xyz format')

    parser = argparse.ArgumentParser(description='Process the input parameters to parse the dataset.')
    parser.add_argument('-split_traces', action="store_true", dest='_split_traces_and_store', help='Flag that tells if we want to create the files to split the traces into train and test.')
    parser.add_argument('-creat_samp_dat', action="store_true", dest='_create_sampled_dataset', help='Flag that tells if we want to create and store the sampled dataset.')
    parser.add_argument('-creat_orig_dat', action="store_true", dest='_create_original_dataset', help='Flag that tells if we want to create and store the original dataset.')
    parser.add_argument('-creat_true_sal', action="store_true", dest='_create_true_saliency', help='Flag that tells if we want to create and store the ground truth saliency.')
    parser.add_argument('-compare_traces', action="store_true", dest='_compare_traces', help='Flag that tells if we want to compare the original traces with the sampled traces.')
    parser.add_argument('-plot_3d_traces', action="store_true", dest='_plot_3d_traces', help='Flag that tells if we want to plot the traces in the unit sphere.')
    parser.add_argument('-creat_synth_dat', action="store_true", dest='_create_synth_egs')
    parser.add_argument('-creat_synth_test', action="store_true", dest='_create_synth_test')
    parser.add_argument('-plot_synthetic_traces', action="store_true", dest='_plot_synth_traces', help='Flag that tells if we want to plot the traces in the unit sphere.')



    args = parser.parse_args()

    if args._split_traces_and_store:
        split_traces_and_store()

    # create_and_store_original_dataset()

    if args._create_sampled_dataset:
        create_and_store_sampled_dataset()

    if args._create_original_dataset:
        create_and_store_original_dataset()

    if args._compare_traces:
        original_dataset = get_original_dataset()
        sampled_dataset = load_sampled_dataset()
        compare_sample_vs_original(original_dataset, sampled_dataset)

    if args._plot_3d_traces:
        sampled_dataset = load_sampled_dataset()
        plot_all_traces_in_3d(sampled_dataset)
        
    if args._create_synth_egs:
        sampled_dataset = load_sampled_dataset()
        create_simple_synthetic_training_set(sampled_dataset)

    if args._create_synth_test:
        sampled_dataset = load_sampled_dataset()
        create_synthetic_test_set(sampled_dataset)
        
    if args._plot_synth_traces:
        sampled_dataset = load_synthetic_dataset()
        plot_all_traces_in_3d(sampled_dataset)  

    if args._create_true_saliency:
        if os.path.isdir(OUTPUT_FOLDER):
            sampled_dataset = load_sampled_dataset()
            create_and_store_true_saliency(sampled_dataset)
        else:
            print('Please verify that the sampled dataset has been created correctly under the folder', OUTPUT_FOLDER)

    # sampled_dataset = load_sampled_dataset()
    # most_salient_point_baseline(sampled_dataset)
