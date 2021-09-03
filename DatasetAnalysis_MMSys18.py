import os
import pandas as pd
import numpy as np
from math import pi
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

## For the dataset of MMSys18
dataFolder = './MMSys18/scanpaths'
alt_dataFolder = './Xu_CVPR_18/dataset/Gaze_txt_files'
videos = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight', '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows',
          '11_Abbottsford', '12_TeatroRegioTorino', '13_Fountain', '14_Warship',
          '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']
test_videos = ['1_PortoRiverside', '3_PlanEnergyBioLab', '5_Waterpark', '14_Warship', '16_Turtle']
test_users = [27,16,21,56,24,5,52,6,11,0,7,55,38,50,46,26,42,54,8,51,14,28,39,23,19,3,25,4,47]
iD_users = [37, 44, 13, 49, 31]
iD_videos = ['2_Diner', '4_Ocean']
sig_vidusers = [('1_PortoRiverside', 4), ('1_PortoRiverside', 19),
                ('1_PortoRiverside', 21), ('1_PortoRiverside', 24),
                ('1_PortoRiverside', 26), ('1_PortoRiverside', 47),
                ('1_PortoRiverside', 54), ('1_PortoRiverside', 55),
                ('3_PlanEnergyBioLab', 3), ('3_PlanEnergyBioLab', 19),
                ('3_PlanEnergyBioLab', 21), ('3_PlanEnergyBioLab', 38),
                ('3_PlanEnergyBioLab', 46), ('3_PlanEnergyBioLab', 50),
                ('3_PlanEnergyBioLab', 51), ('3_PlanEnergyBioLab', 55),
                ('3_PlanEnergyBioLab', 56), ('5_Waterpark', 4),
                ('5_Waterpark', 24), ('5_Waterpark', 51),
                ('14_Warship', 11), ('14_Warship', 24),
                ('14_Warship', 28), ('14_Warship', 42),
                ('14_Warship', 47), ('16_Turtle', 23),
                ('16_Turtle', 26), ('16_Turtle', 39)]
sig_vidusers = {}
sig_vidusers['1_PortoRiverside'] = [4,19,21,24,26,47,54,55]
sig_vidusers['3_PlanEnergyBioLab'] = [3,19,21,38,46,50,51,55,56]
sig_vidusers['5_Waterpark'] = [4,24,51]
sig_vidusers['14_Warship'] = [11,24,28,42,47]
sig_vidusers['16_Turtle'] = [23,26,39]


cases = ['video','user','viduser','alt_dataset']
##cases = ['sig_vidusers']
#videos_ids = np.arange(0, len(videos))
videos_ids = [0,1,2,3,4,13,15]
users = np.arange(0, 57)

# Select at random the users for each set
np.random.shuffle(users)
num_train_users = int(len(users)*0.5)
users_train = users[:num_train_users]
users_test = users[num_train_users:]

# Select at random the videos for each set
np.random.shuffle(videos_ids)
num_train_videos = int(len(videos_ids)*0.8)
videos_ids_train = videos_ids[:num_train_videos]
videos_ids_test = videos_ids[num_train_videos:]

'''
Return longitude \in {0, 1} and latitude \in {0, 1}
'''
def get_positions_for_trace(video, user):
    foldername = os.path.join(dataFolder, video + '_fixations')
    filename = os.path.join(foldername, video + '_fixations_user_' + str(user) + '.csv')
    dataframe = pd.read_table(filename, header=None, sep=",", engine='python')
    dataframe_values = np.array(dataframe.values[:, 0:2])
    return dataframe_values

# Compute the sphere distance with the unit directional vectors in cartesian coordinate system
def compute_orth_dist_with_unit_dir_vec(position_a, position_b):
    yaw_true = (position_a[:, 0] - 0.5) * 2 * pi
    pitch_true = (position_a[:, 1] - 0.5) * pi
    # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
    yaw_pred = (position_b[:, 0] - 0.5) * 2 * pi
    pitch_pred = (position_b[:, 1] - 0.5) * pi
    # Transform into directional vector in Cartesian Coordinate System
    x_true = np.sin(yaw_true)*np.cos(pitch_true)
    y_true = np.sin(pitch_true)
    z_true = np.cos(yaw_true)*np.cos(pitch_true)
    x_pred = np.sin(yaw_pred)*np.cos(pitch_pred)
    y_pred = np.sin(pitch_pred)
    z_pred = np.cos(yaw_pred)*np.cos(pitch_pred)
    # Finally compute orthodromic distance
    # great_circle_distance = np.arccos(x_true*x_pred+y_true*y_pred+z_true*z_pred)
    # To keep the values in bound between -1 and 1
    great_circle_distance = np.arccos(np.maximum(np.minimum(x_true * x_pred + y_true * y_pred + z_true * z_pred, 1.0), -1.0))
    return great_circle_distance
xu_users = range(1, 12)
xu_videos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
             90, 91, 92, 93, 94, 95, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
             96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 131, 132, 133, 134, 135, 136, 137, 138, 139,
             208, 209, 210, 211, 212, 213, 214, 215]
xu_videos_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202]
xu_videos_test = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213, 214, 215]

xu_traces_all = {}
xu_traces_train = {}
xu_traces_test = {}
for user in xu_users:
    xu_traces_all[user] = []
    xu_traces_train[user] = []
    xu_traces_test[user] = []
    foldername = '%s/p%03d' % (alt_dataFolder, user)
    list_videos = os.listdir(foldername)
    for video in list_videos:
        video_id = int(video.split('.')[0])
        xu_traces_all[user].append(video)
        if video_id in xu_videos_test:
            xu_traces_test[user].append(video)
        if video_id in xu_videos_train:
            xu_traces_train[user].append(video)

def get_trace(person, video):
    filename = '%s/p%03d/%s' % (alt_dataFolder, person, video)
    dataframe = pd.read_table(filename, header=None, sep=",", engine='python')
    # The columns in the file that correspond to the head are 3 and 4
    head_values = np.array(dataframe.values[:, 3:5])
    return head_values


n_window = 1
OoD_case_velocities = {}
iD_case_velocities = {}
##for time_t in [0.2, 0.5, 1, 2, 5, 15]:
for case in cases:
    OoD_case_velocities[case] = []
    iD_case_velocities[case] = []
    if case == 'alt_dataset':
        n_window = 1
        for m_window in [25]:
            print ('m_window', m_window * 1.0 / 25.0)
            xu_average_velocities = []
            for user in xu_traces_all.keys():
                print(user)
                for video in xu_traces_all[user]:
                    # print m_window, video, user
                    positions = get_trace(user, video)
                    for x_i in range(n_window, len(positions) - m_window):
                        # This one computes the farthest motion from the last position
                        av_vel = np.max(compute_orth_dist_with_unit_dir_vec(np.array(positions[x_i:x_i + m_window], dtype=np.float32), np.array(np.repeat(positions[x_i - 1:x_i], m_window, axis=0), dtype=np.float32)))
                        # This one computes the additive motion
                        # av_vel = np.sum(compute_orth_dist_with_unit_dir_vec(np.array(positions[x_i:x_i + m_window], dtype=np.float32), np.array(positions[x_i - 1:x_i + m_window - 1], dtype=np.float32)))
                        OoD_case_velocities[case].append(av_vel)
    iD_video = False
    iD_user = False
    sig_video = False
    sig_user = False
    for time_t in [1]:
        print ('time_t', time_t)
        m_window = int(round(time_t * 5.0))
        average_velocities = []
        for video_id in videos_ids:
            video = videos[video_id]
            if video in sig_vidusers.keys() and case == 'sig_vidusers':
                sig_video = True
            if video in iD_videos and case not in ['video','viduser']:
                continue;
            elif video in iD_videos:
                iD_video = True
            else:
                iD_video = False
            velocities_per_video = []
            for user in users:
                if sig_video and user in sig_vidusers[video]:
                    sig_user = True
                if user in iD_users and case not in ['user', 'viduser']:
                    continue;
                elif user in iD_users:
                    iD_user = True
                else:
                    iD_user = False
                # print video, user
                velocities_for_trace = []
                positions = get_positions_for_trace(video, user)
                for x_i in range(n_window, len(positions)-m_window):
                    # This one computes the farthest motion from the last position
                    av_vel = np.max(compute_orth_dist_with_unit_dir_vec(positions[x_i:x_i + m_window], np.repeat(positions[x_i - 1:x_i], m_window, axis=0)))

                    if case == 'sig_vidusers' and sig_video and sig_user:
                        sig_video = False
                        sig_user = False
                        iD_case_velocities[case].append(av_vel)
                    elif case == 'sig_vidusers' and not sig_video and not sig_user:
                        print("NOT SIG")
                        OoD_case_velocities[case].append(av_vel)
                        
                    # video OoD set cases 
                    if case == 'video' and iD_video:
                        if user in test_users:
                            iD_case_velocities[case].append(av_vel)
                    elif case == 'video':
                        if user in test_users:
                            OoD_case_velocities[case].append(av_vel)
                            
                    # user OoD set cases
                    if case == 'user' and iD_user:
                        if video in test_videos:
                            iD_case_velocities[case].append(av_vel)
                    elif case == 'user':
                        if video in test_videos:
                            OoD_case_velocities[case].append(av_vel)
                            
                    # viduser OoD set cases
                    if case == 'viduser':
                        if iD_user and iD_video:
                            iD_case_velocities[case].append(av_vel)
                        elif video in test_videos:
                            OoD_case_velocities[case].append(av_vel)
                    if case == 'alt_dataset':
                        if video in test_videos and user in test_users:
                            iD_case_velocities[case].append(av_vel)
##                    print(av_vel/np.pi)
                    # This one computes the additive motion
                    # av_vel = np.sum(compute_orth_dist_with_unit_dir_vec(positions[x_i:x_i + m_window], positions[x_i - 1:x_i + m_window - 1]))
                    average_velocities.append(av_vel)
                    velocities_for_trace.append(av_vel)
                    velocities_per_video.append(av_vel)
            #print(case, iD_case_velocities[case])
        average_velocities = np.array(average_velocities) * 180 / pi
        # Compute the CDF
        #n, bins, patches = plt.hist(average_velocities, bins=np.arange(0, 360), density=True, histtype='step', cumulative=True, label=str(time_t)+'s')

x = np.arange(len(cases))
barWidth=0.1
iD_velocities = [np.mean(iD_case_velocities[case]) for case in cases]
OoD_velocities = [np.mean(OoD_case_velocities[case]) for case in cases]
print(OoD_velocities)
plt.rcParams["font.family"] = "serif"
#plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots()
r1 = np.arange(len(cases))
r2 = [n + barWidth for n in r1]
r3 = [n + barWidth for n in r2]
iD = ax.bar(r1, np.array(iD_velocities), width=barWidth, edgecolor='white', label='iD')
OoD = ax.bar(r2, np.array(OoD_velocities), width=barWidth, edgecolor='white', label='OoD')
# Add xticks on the middle of the group bars
##ax.set_xlabel('Case', fontweight='bold')
ax.set_ylabel('Angular Velocity (rad/s)')
ax.bar_label(iD, padding=3)
ax.bar_label(OoD, padding=3)
ax.set_yscale("log")
##ax.set_xticks([])
ax.set_xticks([r + barWidth for r in range(len(cases))])
ax.set_xticklabels(['Videos', 'Users', 'Videos & Users', 'True OOD'])
#ax.bar_label([r + barWidth for r in range(len(cases))], ['Videos', 'Users', 'Videos & Users'])#, 'True OOD'])
##ax.set_yticks([0,np.pi/8, np.pi/6,np.pi/4,np.pi/2,np.pi])
#ax.set_yticks(np.pi)
#ax.set_yticklabels(['0','1','2','3','4', '5'])
#ax.set_ytick_labels([r'$\pi$'])
#ax.set_yticklabels(['0',r'$\frac{\pi}{8}$',r'$\frac{\pi}{6}$', r'$\frac{\pi}{4}$', r'\displaystyle $\frac{\pi}{2$}',r'$\pi$'])        
ax.legend()
plt.show()
##plt.xlabel('Motion from last position (t -> t+T) [Degrees]')
##plt.ylabel('Data proportion')
##plt.legend()
##plt.xlim(0, 180)
##plt.ylim(0.0, 1.0)
##plt.title('MMSys 18')
##plt.savefig('cdf_MMSys18.pdf')
plt.show()
