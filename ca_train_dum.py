import sys
sys.path.insert(0, './')

import pickle
import numpy as np
from tensorflow import keras
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.activations import softplus
import tensorflow_probability as tfp
from SampledDataset import read_sampled_positions_for_trace, load_saliency, load_true_saliency, get_video_ids, get_user_ids, get_users_per_video, split_list_by_percentage, partition_in_train_and_test_without_any_intersection, partition_in_train_and_test_without_video_intersection, partition_in_train_and_test
from TRACK_model import create_TRACK_model
from UQ_TRACK_VIB_mlp import create_TRACK_VIB_model
from CVPR18_model import create_CVPR18_model
import MM18_model
from position_only_VIB import create_pos_only_model_7
from Pos_Only_Baseline_3dLoss import Pos_Only_Class
from CVPR18_original_model import create_CVPR18_orig_Model, auto_regressive_prediction
#from Xu_CVPR_18.Read_Dataset import get_videos_train_and_test_from_file
import os
from Utils import cartesian_to_eulerian, eulerian_to_cartesian, get_max_sal_pos, load_dict_from_csv, all_metrics
import argparse
from Saliency_only_baseline import get_most_salient_points_per_video, predict_most_salient_point
from TRACK_AblatSal_model import create_TRACK_AblatSal_model
from TRACK_AblatFuse_model import create_TRACK_AblatFuse_model
from ContentBased_Saliency_baseline import get_most_salient_content_based_points_per_video, predict_most_salient_cb_point


parser.add_argument('-gpu_id', action='store', dest='gpu_id', help='The gpu used to train this network.')
parser.add_argument('-dataset_name', action='store', dest='dataset_name', help='The name of the dataset used to train this network.')
parser.add_argument('-model_name', action='store', dest='model_name', help='The name of the model used to reference the network structure used.')
parser.add_argument('-init_window', action='store', dest='init_window', help='(Optional) Initial buffer window (to avoid stationary part).', type=int)
parser.add_argument('-m_window', action='store', dest='m_window', help='Past history window.', type=int)
parser.add_argument('-h_window', action='store', dest='h_window', help='Prediction window.', type=int)
parser.add_argument('-end_window', action='store', dest='end_window', help='(Optional) Final buffer (to avoid having samples with less outputs).', type=int)
parser.add_argument('-exp_folder', action='store', dest='exp_folder', help='Used when the dataset folder of the experiment is different than the default dataset.')
parser.add_argument('-provided_videos', action="store_true", dest='provided_videos', help='Flag that tells whether the list of videos is provided in a global variable.')
parser.add_argument('-use_true_saliency', action="store_true", dest='use_true_saliency', help='Flag that tells whether to use true saliency (if not set, then the content based saliency is used).')
parser.add_argument('-num_of_peaks', action="store", dest='num_of_peaks', help='Value used to get number of peaks from the true_saliency baseline.')
parser.add_argument('-video_test_chinacom', action="store", dest='video_test_chinacom', help='Which video will be used to test in ChinaCom, the rest of the videos will be used to train')
parser.add_argument('-metric', action="store", dest='metric', help='Which metric to use, by default, orthodromic distance is used.')

args = parser.parse_args()

gpu_id = args.gpu_id
dataset_name = args.dataset_name
model_name = args.model_name
# Buffer window in timesteps
M_WINDOW = args.m_window
# Forecast window in timesteps (5 timesteps = 1 second) (Used in the network to predict)
H_WINDOW = args.h_window
# Initial buffer (to avoid stationary part)
if args.init_window is None:
    INIT_WINDOW = M_WINDOW
else:
    INIT_WINDOW = args.init_window
# final buffer (to avoid having samples with less outputs)
if args.end_window is None:
    END_WINDOW = H_WINDOW
else:
    END_WINDOW = args.end_window
# This variable is used when we use the dataset of the experiment run in the respective paper
# e.g. Xu_CVPR_18 predicts the gaze using the last 5 gaze positions to predict the next 5 gaze positions
EXP_FOLDER = args.exp_folder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

assert dataset_name in ['Xu_PAMI_18', 'AVTrack360', 'Xu_CVPR_18', 'Fan_NOSSDAV_17', 'Nguyen_MM_18', 'David_MMSys_18', 'Li_ChinaCom_18']
assert model_name in ['TRACK', 'CVPR18', 'pos_only', 'no_motion', 'most_salient_point', 'true_saliency', 'content_based_saliency', 'CVPR18_orig', 'TRACK_AblatSal', 'TRACK_AblatFuse', 'MM18', 'pos_only_3d_loss']

NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216

NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

RATE = 0.2

root_dataset_folder = os.path.join('./', dataset_name)

EXP_NAME = '_Paper_Exp_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, EXP_FOLDER)

