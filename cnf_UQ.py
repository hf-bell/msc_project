# code adapted from postels et al., 2021. 
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy as sp
import pandas as pd
import pickle
import sklearn
import sklearn.model_selection
import scipy.special

from tqdm.keras import TqdmCallback
from sklearn import preprocessing
from sklearn import gaussian_process as skgp
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
     import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
 
import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from tqdm import tqdm
from arg_extractor import get_args


# TODO: - code for getting intermediate layer outputs / importing for fitting flow
#       - make this into self-contained unit? i.e. create_flow() & train_flow()? 

def load_embeddings():
    args = get_args()
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
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    root_dataset_folder = os.path.join('./', dataset_name)
    
    # If EXP_FOLDER is defined, add "Paper_exp" to the results name and use the folder in EXP_FOLDER as dataset folder
    if EXP_FOLDER is None:
        EXP_NAME = '_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
        SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, 'sampled_dataset')
    else:
        EXP_NAME = '_Paper_Exp_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
        SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, EXP_FOLDER)

    SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'extract_saliency/saliency')
    TRUE_SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'true_saliency')
    if model_name == 'TRACK_VIB':
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_VIB/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_VIB/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
    elif model_name == 'pos_only':
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'pos_only_UQ/Results_EncDec_eulerian' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'pos_only_UQ/Models_EncDec_eulerian' + EXP_NAME)

    outs = pickle.load(open(RESULTS_FOLDER+'train_layer_acts.pkl', 'rb'))
    return outs

    

def subnet_fc(c_in, c_out):
    # change size of layers? 
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512,  c_out))

outs_dict = load_embeddings()
h_embeddings = outs_dict['h_embeddings']
c_embeddings = outs_dict['c_embeddings']
train_regression_preds = outs_dict['preds']
# TODO: condition dims? Have put 2 here: corresponds to pitch & yaw (for pos-only)
cond = Ff.ConditionNode(2, name='condition')
# shape[1] gives size of embeddings (e.g. in our case 512) 
nodes = [Ff.InputNode(h_embeddings.shape[1], name='input')]

# 4 GLOW coupling blocks
# what does 'clamp' do?
for k in range(4):
    nodes.append(Ff.Node(nodes[-1],
                         Fm.GLOWCouplingBlock,
                         {'subnet_constructor':subnet_fc, 'clamp':2.0},
                         conditions=cond,
                         name=F'coupling_{k}'))
    nodes.append(Ff.Node(nodes[-1],
                         Fm.PermuteRandom,
                         {'seed':k},
                         name=F'permute_{k}'))

nodes.append(Ff.OutputNode(nodes[-1], name='output'))
cinn = Ff.ReversibleGraphNet(nodes + [cond])
# use gpu instead here?
cinn.cpu()



# Train flow using pytorch 
trainable_parameters = [p for p in cinn.parameters() if p.requires_grad]
for p in trainable_parameters:
    p.data = 0.01 * torch.randn_like(p)

optimizer = torch.optim.Adam(trainable_parameters, lr=1e-4, weight_decay=1e-5)

# train_regression_preds[i,0]??? Our outs are 2D, so might be different 
trainloader = torch.utils.data.DataLoader(
    [[h_embeddings[i] + 1e-8 * np.random.normal(*h_embeddings[0].shape), train_regression_preds[i]]
     for i in range(h_embeddings.shape[0])], 
    shuffle=True, batch_size=100)

# need to choose a value instead of 20 for size of validation set. 
flow_x_val = torch.from_numpy(h_embeddings[:600].astype('float32'))
flow_y_val = torch.from_numpy(train_regression_prediction[:600, 0].astype('float32'))
flow_y_val.unsqueeze_(-1)

N_epochs = 1500
nll_mean = []

train_nll = []
val_nll = []
for epoch in tqdm(range(N_epochs)):
    for i, (batch_x, batch_y) in enumerate(trainloader):
        batch_x = torch.as_tensor(batch_x, dtype=torch.float)
        batch_y = torch.tensor(batch_y, dtype=torch.float)
        batch_y.unsqueeze_(-1)
        z, log_j = cinn.forward(batch_x, c=batch_y)

        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / embedding.shape[1]
        nll.backward()
        nn.utils.clip_grad_norm_(trainable_parameters, 10.)
        nll_mean.append(nll.item())
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        z, log_j = cinn(flow_x_val, c=flow_y_val)
        nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / embedding.shape[1]

    train_nll.append(np.mean(nll_mean))
    val_nll.append(nll_val.item())
    nll_mean = []
    #scheduler.step()

path = MODELS_FOLDER+'cnf_model.pt'
torch.save(cinn, path)
plt.figure()
plt.plot(range(len(train_nll)), train_nll, label='training')
plt.plot(range(len(val_nll)), val_nll, label='validation')
plt.legend()
plt.ylim(-10, 1)
plt.savefig(MODELS_FOLDER+'flow_train_val_loss.png')
flow_x = torch.from_numpy(embedding_all.astype('float32'))
