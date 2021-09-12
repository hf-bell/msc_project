import sys
sys.path.insert(0, './')
import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rcParams.update({
##    "text.usetex": True,
    "font.family": "serif",
##    "font.serif": ["Palatino"],
})
dset = 'sstd'
SD = 0.1
gt = 0
lt = 30
pos = 2
case = 'ang'
h_or_c = 'h'
#import seaborn as sns
# code to get embeddings from file
def load_embeddings():
    if dset  =='MMSys_18':
        if case not in ['user', 'viduser']:
            h_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/results_case_%s/h_samples.pkl'%case
            c_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/results_case_%s/c_samples.pkl'%case
        elif case == 'alt_dataset':
            h_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/results_case_%s/h_samples.pkl'%case
            c_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/results_case_%s/c_samples.pkl'%case
        else:
            h_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/results_case_%s/h_samples_user.pkl'%case
            c_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/results_case_%s/c_samples_user.pkl'%case
    elif dset == 'sstd':
        h_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/synthetic_dataset/simple_%s/h_samples.pkl'%(SD)
        c_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/synthetic_dataset/simple_%s/c_samples.pkl'%(SD)
    elif dset == 'hstd':
        if case == 'speed':
            h_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/hstd/%s_pos/h_samples.pkl'%pos
            c_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/hstd/%s_pos/c_samples.pkl'%pos
        else:
            h_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/hstd/gt%s_lt%s/h_samples.pkl'%(gt,lt)
            c_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/hstd/gt%s_lt%s/c_samples.pkl'%(gt,lt)
    #h_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/synthetic_dataset/simple_%s/h_samples.pkl'%(SD)
    #h_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/hstd/gt%s_lt%s/h_samples.pkl'%(gt,lt)
    #h_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/hstd/%s_pos/h_samples.pkl'%pos
    

    #OOD_embeddings = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/OOD_embeddings.pkl'
    #c_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/synthetic_dataset/simple_%s/c_samples.pkl'%(SD)

    # HSTD VIS
    #c_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/hstd/gt%s_lt%s/c_samples.pkl'%(gt,lt)
    #c_embeddings_path = 'David_MMSys_18/pos_only_vary_mix/Models_EncDec_eulerian_Paper_Exp_init_30_in_5_out_25_end_25/hstd/%s_pos/c_samples.pkl'%pos
    with open(h_embeddings_path, "rb") as f:
        h_embeddings = pickle.load(f)
    with open(c_embeddings_path, "rb") as f:
        c_embeddings = pickle.load(f)
    return h_embeddings, c_embeddings

h_data, c_data = load_embeddings()

#for vid in h_data.keys():
    #print(vid, ";",len(h_data[vid]))
if case != 'alt_dataset':
    iD_vids = ['2_Diner','4_Ocean']    
    iD_users = [37, 44, 13, 49, 31]
else:
    print(case)
    iD_vids = ['2_Diner','4_Ocean']
    OoD_vids = ['023']
#data['h_embeddings'].append(OOD_data)
#data = np.squeeze(np.array(data['h_embeddings']))
#print(np.array(OOD_data).shape)
#OOD_data = np.squeeze(np.array(OOD_data))
#h_data_iD = np.array(c_data['simple_iD'])
#h_data_OoD = np.array(c_data['simple_%s'%(SD)])
#[print("vid", vid, len(h_data[vid])) for vid in h_data.keys()]
#h_data_iD = np.array(h_data['iD_traj_50_60'])
#h_data_OoD = np.array(c_data['gt%s_lt%s_50_60'%(gt,lt)])
#h_data_OoD = np.array(h_data['%s_pos_50_60'%(pos)])
if dset == 'sstd':
    if h_or_c == 'c':
        h_data_iD = np.array(c_data['simple_iD'])
        h_data_OoD = np.array(c_data['simple_%s'%(SD)])
    else:
        h_data_iD = np.array(h_data['simple_iD'])
        h_data_OoD = np.array(h_data['simple_%s'%(SD)])
elif dset == 'hstd':
    h_data_iD = np.array(h_data['iD_traj_50_60'])
    if case == 'speed':
        h_data_OoD = np.array(h_data['%s_pos_50_60'%(pos)])
    else:
        h_data_OoD = np.array(c_data['gt%s_lt%s_50_60'%(gt,lt)])
    #h_data_OoD = np.array(c_data['gt%s_lt%s_50_60'%(gt,lt)])
    #h_data_OoD = np.array(h_data['%s_pos_50_60'%(pos)])
elif dset == 'MMSys_18':
    if case == 'video':
        h_data_iD = np.concatenate([np.array(c_data[vid]) for vid in h_data.keys() if vid not in iD_vids], axis=0)
        h_data_OoD = np.concatenate([np.array(c_data[vid]) for vid in h_data.keys() if vid  in iD_vids], axis=0)
    elif case == 'user':
        [[print(np.array(c_data[vid][user]).shape) for user in h_data[vid].keys() if int(user) in iD_users] for vid in h_data.keys()]
        h_data_OoD = np.concatenate([np.concatenate([np.array(h_data[vid][user]) for user in h_data[vid].keys() if int(user) not in iD_users],axis=0) for vid in h_data.keys()], axis=0)
        h_data_iD = np.concatenate([np.concatenate([np.array(h_data[vid][user]) for user in h_data[vid].keys() if int(user) in iD_users],axis=0) for vid in h_data.keys()], axis=0)
    elif case =='viduser':
        h_data_OoD = np.concatenate([np.concatenate([np.array(h_data[vid][user]) for user in h_data[vid].keys() if int(user) not in iD_users],axis=0) for vid in h_data.keys() if vid not in iD_vids], axis=0)
        h_data_iD = np.concatenate([np.concatenate([np.array(h_data[vid][user]) for user in h_data[vid].keys() if int(user) in iD_users],axis=0) for vid in h_data.keys() if vid in iD_vids], axis=0)
    elif case == 'alt_dataset':
    #[print(vid) for vid in h_data.keys() if str(vid) not in OoD_vids]
        h_data_OoD = np.concatenate([np.array(c_data[vid]) for vid in h_data.keys() if str(vid) in OoD_vids],axis=0)
        h_data_iD = np.concatenate([np.array(c_data[vid]) for vid in h_data.keys() if str(vid) in iD_vids],axis=0)
print("iD",h_data_iD.shape)
print("OoD",h_data_OoD.shape)
data = np.concatenate((h_data_OoD, h_data_iD), axis=0)
#OOD_data = np.squeeze(np.array(OOD_data))
#print(data[:-15].shape, data[-15:].shape)
pca = PCA(n_components = 50)

data_reduced = pca.fit_transform(data)
#OOD_data_reduced = pca.transform(OOD_data)
tsne = TSNE(n_components = 2, verbose=1, perplexity=100, n_iter=1000)
data_tsne = tsne.fit_transform(data_reduced)

#OOD_data_tsne = tsne.fit_transform(OOD_data_reduced)
data_2d_one = data_tsne[:,0]
data_2d_two = data_tsne[:,1]
#plt.style.use('fivethirtyeight')
# HSTD scatter use 22500 for SSTD
#plt.scatter(data_tsne[:-9000,0], data_tsne[:-9000,1], marker='.', label='iD')
#plt.scatter(data_tsne[-9000:,0], data_tsne[-9000:,1], marker='.', color='r', label='OoD')
if dset == 'sstd':
    plt.scatter(data_tsne[:-22500,0], data_tsne[:-22500,1], marker='.', color='b',label='OoD', s=2)
    plt.scatter(data_tsne[-22500:,0], data_tsne[-22500:,1], marker='.', color='r', label='iD', s=2)
elif dset == 'hstd':
    plt.scatter(data_tsne[:-9000,0], data_tsne[:-9000,1], marker='.', label='OoD', s=2)
    plt.scatter(data_tsne[-9000:,0], data_tsne[-9000:,1], marker='.', color='r', label='iD', s=2)
elif dset == 'MMSys_18':
# David-MMSys18 scatter
    if case == 'video':
        plt.scatter(data_tsne[:-2610,0], data_tsne[:-2610,1], marker='.', label='iD')
        plt.scatter(data_tsne[-2610:,0], data_tsne[-2610:,1], marker='.', color='r', label='OoD')
    elif case == 'user':
        plt.scatter(data_tsne[:-1125,0], data_tsne[:-1125,1], marker='.', label='OoD')
        plt.scatter(data_tsne[-1125:,0], data_tsne[-1125:,1], marker='.', color='r', label='iD')
    elif case == 'viduser':
        plt.scatter(data_tsne[:-450,0], data_tsne[:-450,1], marker='.', label='OoD')
        plt.scatter(data_tsne[-450:,0], data_tsne[-450:,1], marker='.', color='r', label='iD')
    elif case == 'alt_dataset':
        plt.scatter(data_tsne[:-2520,0], data_tsne[:-2520,1], marker='.', label='OoD')
        plt.scatter(data_tsne[-2520:,0], data_tsne[-2520:,1], marker='.', color='r', label='iD')
#plt.xscale("log")
#plt.scatter(OOD_data_tsne[:-15,1],OOD_data_tsne[:-15,1], marker = '*', color='r')
plt.legend()
if dset=='sstd':
    plt.savefig('sstd_features/sstd_%s_%s.pdf'%(SD, h_or_c))
elif dset=='hstd':
    plt.savefig('hstd_features/hstd_features_gt%s__lt%s_%s.pdf'%(gt,lt, h_or_c))
elif dset=='MMSys_18':
    plt.savefig('MMSys18_features/MMSys18_features_case_%s_c.pdf'%case)
