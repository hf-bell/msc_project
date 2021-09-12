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

SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'extract_saliency/saliency')

if model_name == 'TRACK_VIB':
    RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_VIB/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_VIB/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
elif model_name == 'pos_only':
    RESULTS_FOLDER = os.path.join(root_dataset_folder, 'pos_only/Results_EncDec_eulerian' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, 'pos_only/Models_EncDec_eulerian' + EXP_NAME)    

PERC_VIDEOS_TRAIN = 0.8
PERC_USERS_TRAIN = 0.5PERC_VIDEOS_TRAIN = 0.8
PERC_USERS_TRAIN = 0.5

if args.provided_videos:
    if dataset_name == 'Xu_CVPR_18':
        videos_train, videos_test = get_videos_train_and_test_from_file(root_dataset_folder)
        partition = partition_in_train_and_test_without_video_intersection(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, videos_train, videos_test, users_per_video)
    elif dataset_name == 'David_MMSys_18':
        train_traces = load_dict_from_csv(os.path.join(root_dataset_folder, 'train_set'), ['user', 'video'])
        test_traces = load_dict_from_csv(os.path.join(root_dataset_folder, 'test_set'), ['user', 'video'])
        print(train_traces)
        print(test_traces)
        partition = partition_in_train_and_test(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, train_traces, test_traces)
        videos_test = ['1_PortoRiverside', '3_PlanEnergyBioLab', '5_Waterpark', '14_Warship', '16_Turtle']
else:
    videos_train, videos_test = split_list_by_percentage(videos, PERC_VIDEOS_TRAIN)
    users_train, users_test = split_list_by_percentage(users, PERC_USERS_TRAIN)
    # Datasets
    partition = partition_in_train_and_test_without_any_intersection(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, videos_train, videos_test, users_train, users_test)

# Dictionary that stores the traces per video and user
all_traces = {}
for video in videos:
    all_traces[video] = {}
    for user in users_per_video[video]:
        all_traces[video][user] = read_sampled_positions_for_trace(SAMPLED_DATASET_FOLDER, str(video), str(user))

# Load the saliency only if it's not the position_only baseline
if model_name not in ['pos_only', 'pos_only_3d_loss', 'no_motion', 'true_saliency', 'content_based_saliency']:
    all_saliencies = {}
    for video in videos:
        all_saliencies[video] = load_saliency(SALIENCY_FOLDER, video)

def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2*np.pi, np.pi])
    return eulerian_batches

def transform_normalized_eulerian_to_cartesian(positions):
    positions = positions * np.array([2*np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)

def generate_arrays(list_IDs, future_window):
    while True:
        encoder_pos_inputs_for_batch = []
        encoder_sal_inputs_for_batch = []
        decoder_pos_inputs_for_batch = []
        decoder_sal_inputs_for_batch = []
        decoder_outputs_for_batch = []
        count = 0
        np.random.shuffle(list_IDs)
        for ID in list_IDs:
            user = ID['user']
            video = ID['video']
            x_i = ID['time-stamp']
            # Load the data
            if model_name not in ['pos_only', 'pos_only_3d_loss', 'MM18']:
                encoder_sal_inputs_for_batch.append(np.expand_dims(all_saliencies[video][x_i-M_WINDOW+1:x_i+1], axis=-1))
                decoder_sal_inputs_for_batch.append(np.expand_dims(all_saliencies[video][x_i+1:x_i+future_window+1], axis=-1))
            if model_name == 'CVPR18_orig':
                encoder_pos_inputs_for_batch.append(all_traces[video][user][x_i-M_WINDOW+1:x_i+1])
                decoder_outputs_for_batch.append(all_traces[video][user][x_i+1:x_i+1+1])
            elif model_name == 'MM18':
                encoder_sal_inputs_for_batch.append(np.concatenate((all_saliencies[video][x_i-M_WINDOW+1:x_i+1], all_headmaps[video][user][x_i-M_WINDOW+1:x_i+1]), axis=1))
                decoder_outputs_for_batch.append(all_headmaps[video][user][x_i+future_window+1])
            else:
                encoder_pos_inputs_for_batch.append(all_traces[video][user][x_i-M_WINDOW:x_i])
                decoder_pos_inputs_for_batch.append(all_traces[video][user][x_i:x_i+1])
                decoder_outputs_for_batch.append(all_traces[video][user][x_i+1:x_i+future_window+1])
            count += 1
            if count == BATCH_SIZE:
                count = 0
                if model_name in ['TRACK', 'TRACK_VIB', 'TRACK_AblatSal', 'TRACK_AblatFuse']:
                    yield ([np.array(encoder_pos_inputs_for_batch), np.array(encoder_sal_inputs_for_batch), np.array(decoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)], np.array(decoder_outputs_for_batch))
                elif model_name == 'CVPR18':
                    yield ([np.array(encoder_pos_inputs_for_batch), np.array(decoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)], np.array(decoder_outputs_for_batch))
                elif model_name == 'pos_only':
                    yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
                elif model_name in ['pos_only_3d_loss']:
                    yield ([np.array(encoder_pos_inputs_for_batch), np.array(decoder_pos_inputs_for_batch)] , np.array(decoder_outputs_for_batch))
                elif model_name == 'CVPR18_orig':
                    yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)[:, 0, :, :, 0]], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch)[:, 0])
                elif model_name == 'MM18':
                    yield (np.array(encoder_sal_inputs_for_batch), np.array(decoder_outputs_for_batch))
                encoder_pos_inputs_for_batch = []
                encoder_sal_inputs_for_batch = []
                decoder_pos_inputs_for_batch = []
                decoder_sal_inputs_for_batch = []
                decoder_outputs_for_batch = []
        if count != 0:
            if model_name in ['TRACK', 'TRACK_VIB']:
                yield ([np.array(encoder_pos_inputs_for_batch), np.array(encoder_sal_inputs_for_batch), np.array(decoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)], np.array(decoder_outputs_for_batch))
            elif model_name == 'CVPR18':
                yield ([np.array(encoder_pos_inputs_for_batch), np.array(decoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)], np.array(decoder_outputs_for_batch))
            elif model_name == 'pos_only':
                yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch)], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch))
            elif model_name in ['pos_only_3d_loss']:
                yield ([np.array(encoder_pos_inputs_for_batch), np.array(decoder_pos_inputs_for_batch)] , np.array(decoder_outputs_for_batch))
            elif model_name == 'CVPR18_orig':
                yield ([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)[:, 0, :, :, 0]], transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch)[:, 0])
            elif model_name == 'MM18':
                yield (np.array(encoder_sal_inputs_for_batch), np.array(decoder_outputs_for_batch))

if model_name == 'TRACK_VIB':
    model = create_TRACK_VIB_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)
elif model_name == 'pos_only':
    models = create_pos_only_model_7(M_WINDOW, H_WINDOW, BETA_h=1e-5, BETA_c=1e-6)
    model = models['model']
    dist_model = models['dist_model']

if model_name not in ['no_motion', 'most_salient_point', 'true_saliency', 'content_based_saliency', 'MM18']:
        model.load_weights(MODELS_FOLDER + '/weights.hdf5')

traces_count = 0
first = True
for ID in partition['train']:
    traces_count += 1
    print('Progress:', traces_count, '/', len(partition['train']))

    user = ID['user']
    video = ID['video']
    x_i = ID['time-stamp']

    # Load the data
    if model_name not in ['pos_only', 'no_motion', 'true_saliency', 'content_based_saliency', 'pos_only_3d_loss', 'MM18']:
        encoder_sal_inputs_for_sample = np.array([np.expand_dims(all_saliencies[video][x_i - M_WINDOW + 1:x_i + 1], axis=-1)])
        decoder_sal_inputs_for_sample = np.array([np.expand_dims(all_saliencies[video][x_i + 1:x_i + H_WINDOW + 1], axis=-1)])
    else:
        encoder_pos_inputs_for_sample = np.array([all_traces[video][user][x_i-M_WINDOW:x_i]])
        decoder_pos_inputs_for_sample = np.array([all_traces[video][user][x_i:x_i + 1]])

    groundtruth = all_traces[video][user][x_i+1:x_i+H_WINDOW+1]

    if model_name == 'TRACK_VIB':
        model_prediction = model.predict([np.array(encoder_pos_inputs_for_sample), np.array(encoder_sal_inputs_for_sample), np.array(decoder_pos_inputs_for_sample), np.array(decoder_sal_inputs_for_sample)])[0]
        # UQ: model uncertainty predictions using VIB
    elif model_name == 'pos_only':
        mu_h ,sigma_h, mu_c, sigma_c = dist_model.predict([transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_sample), transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_sample)])[0]
        if first:
            mu_h_all = np.array(mu_h)
            sigma_h_all = np.array(sigma_h)
            mu_c_all = np.array(mu_c)
            sigma_c_all = np.array(sigma_c)
        else:
            mu_h_all = np.stack((mu_h_all, mu_h), axis=0)
            sigma_h_all = np.stack((sigma_h_all, sigma_h), axis=0)
            mu_c_all = np.stack((mu_c_all, mu_h), axis=0)
            sigma_c_all = np.stack((sigma_c_all, sigma_c), axis=0)
outs = {'mu_h':mu_h_all, 'sigma_h': sigma_h_all, 'mu_c':mu_c_all, 'sigma_c':sigma_c_all}
pickle.dump(outs, RESULTS_FOLDER+'train_layer_acts.pkl', 'wb')
