import sys
sys.path.insert(0, './')


from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
from sklearn import preprocessing
import matplotlib.pyplot as plt

from Utils import eulerian_to_cartesian, cartesian_to_eulerian
from TRACK_model import create_TRACK_model
from UQ_TRACK_VIB import create_TRACK_VIB_model

import pickle

parser = argparse.ArgumentParser(description='Process the input parameters to evaluate the network.')

parser.add_argument('-gpu_id', action='store', dest='gpu_id', help='The gpu used to train this network.')
parser.add_argument('-model_name', action='store', dest='model_name', default='TRACK_VIB',help='The name of the model used to reference the network structure used.')
parser.add_argument('-init_window', action='store', dest='init_window', help='(Optional) Initial buffer window (to avoid stationary part).', type=int)
parser.add_argument('-m_window', action='store', dest='m_window', help='Past history window.', type=int)
parser.add_argument('-h_window', action='store', dest='h_window', help='Prediction window.', type=int)
parser.add_argument('-dataset_name', action='store', dest='dataset_name', help='Dataset being evaluated upon')
parser.add_argument('-exp_folder', action='store', dest='exp_folder', help='Used when the dataset folder of the experiment is different than the default dataset.')
parser.add_argument('-provided_videos', action="store_true", dest='provided_videos', help='Flag that tells whether the list of videos is provided in a global variable.')

args = parser.parse_args()
sess = tf.compat.v1.InteractiveSession()
dataset_name = args.dataset_name
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
EXP_FOLDER = args.exp_folder


root_dataset_folder = os.path.join('./', dataset_name)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

EPOCHS = 500

NUM_TILES_WIDTH = 384
NUM_TILES_HEIGHT = 216

NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256

RATE = 0.2

# If EXP_FOLDER is defined, add "Paper_exp" to the results name and use the folder in EXP_FOLDER as dataset folder
if EXP_FOLDER is None:
    EXP_NAME = '_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
    SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, 'sampled_dataset')
else:
    EXP_NAME = '_Paper_Exp_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
    SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, EXP_FOLDER)

if model_name == 'TRACK':
    RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Models_EncDec_3DCoords_ContSal' + EXP_NAME)

elif model_name in ['TRACK_VIB', 'UQ_TRACK_VIB']:
    RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_VIB/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_VIB/Models_EncDec_3DCoords_ContSal' + EXP_NAME)

PERC_VIDEOS_TRAIN = 0.8
PERC_USERS_TRAIN = 0.5

BATCH_SIZE = 128.0

videos = get_video_ids(SAMPLED_DATASET_FOLDER)
users = get_user_ids(SAMPLED_DATASET_FOLDER)
users_per_video = get_users_per_video(SAMPLED_DATASET_FOLDER)

if args.provided_videos:
    if dataset_name == 'Xu_CVPR_18':
        videos_train, videos_test = get_videos_train_and_test_from_file(root_dataset_folder)
        partition = partition_in_train_and_test_without_video_intersection(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, videos_train, videos_test, users_per_video)

    elif dataset_name == 'David_MMSys_18':
        train_traces = load_dict_from_csv(os.path.join(root_dataset_folder, 'train_set'), ['user', 'video'])
        test_traces = load_dict_from_csv(os.path.join(root_dataset_folder, 'test_set'), ['user', 'video'])

else:
    videos_train, videos_test = split_list_by_percentage(videos, PERC_VIDEOS_TRAIN)
    users_train, users_test = split_list_by_percentage(users, PERC_USERS_TRAIN)
    # Datasets
    partition = partition_in_train_and_test_without_any_intersection(SAMPLED_DATASET_FOLDER, INIT_WINDOW, END_WINDOW, videos_train, videos_test, users_train, users_test)

# Dictionary that stores the traces per video and user
all_saliencies = {}
all_traces = {}
for video in videos:
    all_traces[video] = {}
    for user in users_per_video[video]:
        all_traces[video][user] = read_sampled_positions_for_trace(SAMPLED_DATASET_FOLDER, str(video), str(user))


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
                encoder_pos_inputs_for_batch = []
                encoder_sal_inputs_for_batch = []
                decoder_pos_inputs_for_batch = []
                decoder_sal_inputs_for_batch = []
                decoder_outputs_for_batch = []
        if count != 0:
            if model_name in ['TRACK', 'TRACK_VIB']:
                yield ([np.array(encoder_pos_inputs_for_batch), np.array(encoder_sal_inputs_for_batch), np.array(decoder_pos_inputs_for_batch), np.array(decoder_sal_inputs_for_batch)], np.array(decoder_outputs_for_batch))

if model_name == 'TRACK':
    model = create_TRACK_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)

elif model_name == 'TRACK_VIB':
    model = create_TRACK_VIB_model(M_WINDOW, H_WINDOW, NUM_TILES_HEIGHT, NUM_TILES_WIDTH)

############################################################################
# EVALUATE
############################################################################
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

for v in tf.compat.v1.trainable_variables():
    if "mu" in v.name:
        mus = v
    elif "rho" in v.name:
        rhos = v
    elif "mix_logits" in v.name:
        mix_logits = v

MIXTURE_FOLDER = MODELS_FOLDER+'/mixture_params'
saver = tf.compat.v1.train.Saver(var_list=[mus,rhos,mix_logits])
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
def degree_distance(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi * 180

# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
def vector_to_ang(_v):
    _v = np.array(_v)
    alpha = degree_distance(_v, [0, 1, 0])#degree between v and [0, 1, 0]
    phi = 90.0 - alpha
    proj1 = [0, np.cos(alpha/180.0 * np.pi), 0] #proj1 is the projection of v onto [0, 1, 0] axis
    proj2 = _v - proj1#proj2 is the projection of v onto the plane([1, 0, 0], [0, 0, 1])
    theta = degree_distance(proj2, [1, 0, 0])#theta = degree between project vector to plane and [1, 0, 0]
    sign = -1.0 if degree_distance(_v, [0, 0, -1]) > 90 else 1.0
    theta = sign * theta
    return theta, phi

# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
def ang_to_geoxy(_theta, _phi, _h, _w):
    x = _h / 2.0 - (_h / 2.0) * np.sin(_phi / 180.0 * np.pi)
    temp = _theta
    if temp < 0: temp = 180 + temp + 180
    temp = 360 - temp
    y = (temp * 1.0 / 360 * _w)
    return int(x), int(y)

# ToDo: copied exactly from https://github.com/phananh1010/PanoSalNet/blob/master/lstm.py
def create_fixation_map(v, cartesian):
    if cartesian:
        theta, phi = vector_to_ang(v)
    else:
        theta = v[0]
        phi = v[1]
    hi, wi = ang_to_geoxy(theta, phi, H, W)
    result = np.zeros(shape=(H, W))
    result[H - hi - 1, W - wi - 1] = 1
    return result


# NOTE: Uses methods from MM18 (see orig. MM18 baselines.py)
gblur_size = 5
def create_head_map(v, cartesian=True):
    headmap = create_fixation_map(v, cartesian)
    headmap = cv2.GaussianBlur(headmap, (gblur_size, gblur_size), 0)
    # To binarize the headmap
    headmap = np.ceil(headmap)
    return headmap

def compute_accuracy_metric(binary_pred, binary_true):
    binary_pred = binary_pred.reshape(-1)
    binary_true = binary_true.reshape(-1)
    sum_of_binary = binary_true + binary_pred
    Intersection = np.sum(sum_of_binary == 2)
    Union = np.sum(sum_of_binary>0)
    return Intersection / np.float(Union)

def transform_the_radians_to_original(yaw, pitch):
    yaw = ((yaw/(2*np.pi))*360.0)-180.0
    pitch = ((pitch/np.pi)*180.0)-90.0
    return yaw, pitch



traces_count = 0
x_over_u_per_vid = {}
x_over_u_per_ts = {}
for ID in partition['test']:
    traces_count += 1
    print('Progress:', traces_count, '/', len(partition['test']))

    user = ID['user']
    video = ID['video']
    x_i = ID['time-stamp']
    encoder_sal_inputs_for_sample = np.array([np.expand_dims(all_saliencies[video][x_i - M_WINDOW + 1:x_i + 1], axis=-1)])
    decoder_sal_inputs_for_sample = np.array([np.expand_dims(all_saliencies[video][x_i + 1:x_i + H_WINDOW + 1], axis=-1)])
    groundtruth = all_traces[video][user][x_i+1:x_i+H_WINDOW+1]
    if model_name == 'TRACK':
        model_prediction = model.predict([np.array(encoder_pos_inputs_for_sample), np.array(encoder_sal_inputs_for_sample), np.array(decoder_pos_inputs_for_sample), np.array(decoder_sal_inputs_for_sample)])[0]
    elif model_name == 'TRACK_VIB':
        model_prediction = model.predict([np.array(encoder_pos_inputs_for_sample),
                                          np.array(encoder_sal_inputs_for_sample), np.array(decoder_pos_inputs_for_sample),
                                          np.array(decoder_sal_inputs_for_sample)])[0]


        # UQ: model uncertainty predictions using VIB

        # Get encoder distribution layer outputs
        enc_layer_name = 'sal_enc_map'
        enc_values_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(enc_layer_name).output)
        enc_output = enc_values_model([np.array(encoder_pos_inputs_for_sample),
                                          np.array(encoder_sal_inputs_for_sample), np.array(decoder_pos_inputs_for_sample),
                                          np.array(decoder_sal_inputs_for_sample)])[0]
        
        for v in tf.compat.v1.trainable_variables():
            if "mu" in v.name:
                mus = v
            elif "rho" in v.name:
                rhos = v
            elif "mix_logits" in v.name:
                mix_logits = v
        
        mus, rhos, mix_logits = sess.run(mus, rhos, mix_logits)
        marginal = marginal_dist(mus, rhos, mix_logits)

    for t in range(len(groundtruth)):
        yaw_pred, pitch_pred = cartesian_to_eulerian(model_prediction[0], model_prediction[1], model_prediction[2])
        yaw_pred, pitch_pred = transform_the_radians_to_original(yaw_pred, pitch_pred)
        pred_head_map = create_head_map(np.array([yaw_pred, pitch_pred]), cartesian=False)
        groundtruth_head_map = create_head_map(groundtruth[t])
        if t not in x_over_u_per_vid[video].keys():
            x_over_u_per_vid[video][t] = []
        x_over_u_per_vid[video][t].append(compute_accuracy_metric(pred_head_map, groundtruth_head_map))
        if t not in x_over_u_per_ts.keys():
            x_over_u_per_ts[t] = []
        x_over_u_per_ts[t].append(compute_accuracy_metric(pred_head_map, groundtruth_head_map))


for t in range(H_WINDOW):
    avg_error_per_timestep = []
    #print('Average', t, np.mean(errors_per_timestep[t]), end=';')
    avg_xou_per_timestep.append(np.mean(x_over_u_per_ts[t]))

plt.plot(np.arange(H_WINDOW)+1*RATE, avg_xou_per_timestep, label=model_name)
plt.xlabel('Prediction step (sec.)')
plt.ylabel('Score (Intersection over union)')
plt.grid()
plt.legend()
plt.savefig('avg_x_over_u_%s.png' % (model_name))

                                              
