import numpy as np
import pickle
import sys
import os
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import analysis_utils as anau
plt.rcParams.update({
##    "text.usetex": True,
    "font.family": "serif",
##    "font.serif": ["Palatino"],
})
case = 'speed'
BETA_experiment = False
model = 'VIB'
BETA_h = 4
BETA_c = 4
SD = 2
gt = 30
lt = 180
pos = 5
iD_vids = ['simple_iD']
##iD_vids = ['iD_traj_50_60']
if model == 'VIB':
    # VIB folder details
    
    OOD_FOLDER = './simple_dataset/200/BETA_1e-2'
##    OOD_FOLDER = './HSTD'
##    results_folder = OOD_FOLDER+'/OOD_5_pos_CNF'
    results_folder = OOD_FOLDER+'/simple_%s_results_%s' %(SD, model)
##    if case == 'ang':
##        results_folder = OOD_FOLDER+'/gt%s_lt%s'%(gt,lt)
##    elif case == 'speed':
##        results_folder = OOD_FOLDER+'/%s_pos'%pos
    rates_h = results_folder + '/per_vid_user_rate_h.pkl'
    rates_c = results_folder + '/per_vid_user_rate_c.pkl'
    with open(rates_h, 'rb') as f:
        rates_h = pickle.load(f)
    with open(rates_c, 'rb') as f:
        rates_c = pickle.load(f)

        ##### Rates computation
    avg_h_per_traj = {}
    avg_c_per_traj = {}
    for vid in rates_h.keys():
        avg_h_per_traj[vid] = []
        avg_c_per_traj[vid] = []
        for user in rates_h[vid].keys():
            avg_h_per_traj[vid].append(np.mean(rates_h[vid][user]))
            avg_c_per_traj[vid].append(np.mean(rates_c[vid][user]))
    h_rates, c_rates, gt_labels, h_dict, c_dict = anau.rates_get_vids_and_compile_labels(
        rates_h, rates_c, iD_vids
        )

    [print("VIDEO %s, AVERAGE H RATE %s, H VARIANCE %s" % (vid, np.mean(avg_h_per_traj[vid]), np.var(avg_h_per_traj[vid]))) for vid in avg_h_per_traj.keys()]
    [print("VIDEO %s, AVERAGE C RATE %s, C VARIANCE %s" % (vid, np.mean(avg_c_per_traj[vid]), np.var(avg_h_per_traj[vid]))) for vid in avg_c_per_traj.keys()]
    
    print("h-state AUROC",metrics.roc_auc_score(gt_labels, h_rates))
    print("c-state AUROC",metrics.roc_auc_score(gt_labels, c_rates))
    plt.scatter(h_dict['iD'], c_dict['iD'], marker='.', color='b',s=2, label='iD')
    plt.scatter(h_dict['OoD'], c_dict['OoD'], marker='.',color='r',s=2, label='OoD')
    plt.legend()
    plt.xlabel('h-state per-instance rate value')
    plt.ylabel('c-state per-instance rate value')
    if case == 'ang':
        plt.title('Distribution of iD vs OoD rate values for HSTD trajectories, gt%s_lt%s'%(gt,lt))
    elif case == 'speed':
        d=1
##        plt.title('Distribution of iD vs OoD rate values for HSTD trajectories, speed=%s'%(pos))   
    plt.show()
    plt.cla()
    fpr, tpr, threshold = metrics.roc_curve(gt_labels, c_rates)
    print(threshold)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'C AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    fpr, tpr, threshold = metrics.roc_curve(gt_labels, h_rates)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'g', label = 'H AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')

    plt.show()


if model == 'CNF':
    OOD_FOLDER = './constrained_synthetic_dataset'
    results_folder = OOD_FOLDER+'/gt70_lt180_OOD_CNF/'
    flows = results_folder+'test_flows.pkl'
    labels = results_folder+'ood_labels.pkl'

    with open(flows, 'rb') as f:
        flows = pickle.load(f)
    with open(labels, 'rb') as f:
        gt_labels = pickle.load(f)
    print("CNF AUROC",metrics.roc_auc_score(gt_labels, flows))

