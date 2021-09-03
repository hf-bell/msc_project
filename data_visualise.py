import os
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

## For the dataset of MMSys18
dataFolder = 'DatasetAnalysis/MMSys18/scanpaths'
videos = ['1_PortoRiverside', '2_Diner', '3_PlanEnergyBioLab', '4_Ocean', '5_Waterpark', '6_DroneFlight', '7_GazaFishermen', '8_Sofa', '9_MattSwift', '10_Cows', '11_Abbottsford', '12_TeatroRegioTorino', '13_Fountain', '14_Warship', '15_Cockpit', '16_Turtle', '17_UnderwaterPark', '18_Bar', '19_Touvet']
videos_ids = np.arange(0, len(videos))
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
