import sys
sys.path.insert(0, './')

import numpy as np
from tensorflow import keras
from SampledDataset import read_sampled_positions_for_trace, load_saliency, load_true_saliency, get_video_ids, get_user_ids, \
     get_users_per_video, split_list_by_percentage, partition_in_train_and_test_without_any_intersection, partition_in_train_and_test_without_video_intersection, partition_in_train_and_test
from TRACK_model import create_TRACK_model
from UQ_TRACK_VIB import create_TRACK_VIB_model
from CVPR18_model import create_CVPR18_model
import MM18_model
from position_only_baseline import create_pos_only_model
from Pos_Only_Baseline_3dLoss import Pos_Only_Class
from CVPR18_original_model import create_CVPR18_orig_Model, auto_regressive_prediction
#from Xu_CVPR_18.Read_Dataset import get_videos_train_and_test_from_file
import os
from Utils import cartesian_to_eulerian, eulerian_to_cartesian, get_max_sal_pos, load_dict_from_csv, all_metrics
import argparse

def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2*np.pi, np.pi])
    return eulerian_batches

def transform_normalized_eulerian_to_cartesian(positions):
    positions = positions * np.array([2*np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)
