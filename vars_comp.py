import numpy as np
import pickle
import sys
import os
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "serif"

from scipy.stats import pearsonr
from scipy.stats import spearmanr
import sklearn.metrics as metrics


scatter_plots_ts = False
case = 'regression'
BETA_experiment = True
model = 'VIB'

# VIB folder details
BETA_h = -5
BETA_c = -6
OOD_FOLDER = './pos-only-VIB/UQ_pos_only_VIB_OOD_200_comps_MMSys18'

if BETA_experiment:
    BETA_vals = 'BETA_h_1e%s__BETA_c_1e%s' % (BETA_h, BETA_c)
    OOD_FOLDER = OOD_FOLDER+'/BETA_experiments/' +BETA_vals

if case is not None:
    results_folder = OOD_FOLDER + '/results_%s'%(case)
else:
    results_folder = './pos-only-VIB/UQ_pos_only_VIB_200_comps_MMSys18'
    if BETA_experiment:
        results_folder += '/BETA_experiments/' +BETA_vals
    
def get_and_compile_global_vars_corr(results_folder):
    pos_only_var_ts = results_folder+'/var_per_ts.pkl'
    err_ts = results_folder+'/err_per_ts.pkl'

    with open(pos_only_var_ts, 'rb') as f:
        pos_only_vars_ts = pickle.load(f)

    with open(err_ts, 'rb') as f:
        pos_only_err_ts = pickle.load(f)

    interval = range(25)
    videos = []
    avg_vars_per_ts = []
    avg_err_per_ts = []
    for t in pos_only_vars_ts.keys():
        print(t)
        avg_vars_per_ts.append(np.mean(pos_only_vars_ts[t]))
        avg_err_per_ts.append(np.mean(pos_only_err_ts[t]))

    pcorr, _ = pearsonr(avg_vars_per_ts, avg_err_per_ts)
    scorr, _ = spearmanr(avg_vars_per_ts, avg_err_per_ts)
    return pcorr, scorr, avg_vars_per_ts, avg_err_per_ts

def get_and_compile_per_step_corr(results_folder):
    pos_only_var_ts = results_folder+'/var_per_ts.pkl'
    err_ts = results_folder+'/err_per_ts.pkl'

    with open(pos_only_var_ts, 'rb') as f:
        pos_only_vars_ts = pickle.load(f)

    with open(err_ts, 'rb') as f:
        pos_only_err_ts = pickle.load(f)

    for t in pos_only_vars_ts.keys():
        plt.style.use('fivethirtyeight')
        plt.scatter(pos_only_err_ts[t],pos_only_vars_ts[t], marker=".", color='r')
        plt.show()
        plt.clf()

    return None

def check_for_var_outliers_align(results_folder):
    var_vids = results_folder + '/var_per_vid.pkl'
    err_vids = results_folder + '/err_per_vid.pkl'
    with open(var_vids, 'rb') as f:
        vars_vids = pickle.load(f)

    with open(err_vids, 'rb') as f:
        err_vids = pickle.load(f)

    interval = range(25)
    videos = []
    avg_vars_per_vid = []
    avg_errs_per_vid = []
    greq_leq = {}
    errs_ts = {}
    # pos_only_vars[video][timestep][vars_for_timestep]
    for k in vars_vids.keys():
        greq_leq[k] = []
        videos.append(k)
        avg_vars_per_ts = []
        avg_err_per_ts = []
        errs_ts[k] = []
        for t in interval:
            avg_vars_per_ts.append(np.mean(vars_vids[k][t]))
            avg_err_per_ts.append(np.mean(err_vids[k][t]))
            thresh = 0.00002
            i = 0
            for elem in vars_vids[k][t]:
                if elem > thresh:
                    greq_leq[k].append(1)
                else:
                    greq_leq[k].append(0)
                errs_ts[k].append(err_vids[k][t][i])
                i += 1
        print("Video",k,"AUROC",metrics.roc_auc_score(greq_leq[k], errs_ts[k]))
        plt.style.use('fivethirtyeight')
        plt.scatter(avg_err_per_ts,avg_vars_per_ts, marker=".", color='r')
        #plt.show()
        #plt.clf()
    #for k in greq:
        #print(k, "ratio", greq[k]/leq[k])
            
        #avg_vars_per_vid.append(np.mean(avg_vars_per_ts))
        #avg_errs_per_vid.append(np.mean(avg_err_per_ts))

def check_for_var_user_outliers(results_folder):
    print("THIS FUNCTION")
    var_vids = results_folder + '/var_per_vid_user.pkl'
    err_vids = results_folder + '/err_per_vid_user.pkl'
    thresholds = [0.00001, 0.00005,0.0001, 0.0008,
                  0.001, 0.002, 0.004, 0.005, 0.008, 0.009,
                  0.01, 0.012, 0.013,0.014, 0.015, 0.018, 0.02, 0.03]
    AUROCS = {}
    with open(var_vids, 'rb') as f:
        vars_vids = pickle.load(f)

    with open(err_vids, 'rb') as f:
        err_vids = pickle.load(f)
    interval = range(25)
    videos = []
    users = []
    sig_v_u = {}
    errs_ts = {}
    vars_ts = {}
    err_vid = {}
    var_vid = {}
    errs_vid_ts = {}
    vars_vid_ts = {}
    greq_leq = {}
    vid_gl_counts = {}
    for v in vars_vids.keys():
        vid_gl_counts[v] = []
        AUROCS[v] = []
        videos.append(v)
        errs_ts[v] = {}
        vars_ts[v] = {}
        errs_vid_ts[v] = []
        err_vid[v] = []
        var_vid[v] = []
        vars_vid_ts[v] = []
##        greq_leq[v] = {}
        greq_leq[v] = {}
        avg_vars_per_ts = []
        avg_errs_per_ts = []
        for u in vars_vids[v].keys():
            errs_ts[v][u] = []
            vars_ts[v][u] = []
##            greq_leq[v][u] = []
            avg_vars_per_user_ts = []
            avg_errs_per_user_ts = []
            users.append(u)
            for t_stamp in range(len(vars_vids[v][u][0])):
                vars_per_t_stamp = []
                errs_per_t_stamp = []
                vars_per_user_t_stamp = []
                errs_per_user_t_stamp = []
##                for t in interval:
##                    vars_per_t_stamp.append(vars_vids[v][u][t][t_stamp])
##                    errs_per_t_stamp.append(err_vids[v][u][t][t_stamp])
##                    vars_per_user_t_stamp.append(vars_vids[v][u][t][t_stamp])
##                    errs_per_user_t_stamp.append(err_vids[v][u][t][t_stamp])
##                avg_vars_per_ts.append(np.mean(vars_per_t_stamp))
##                avg_errs_per_ts.append(np.mean(errs_per_t_stamp))
##                avg_vars_per_user_ts.append(np.mean(vars_per_user_t_stamp))
##                avg_errs_per_user_ts.append(np.mean(errs_per_user_t_stamp))

            for t in interval:
                avg_vars_per_ts.append(np.mean(vars_vids[v][u][t]))
                avg_errs_per_ts.append(np.mean(err_vids[v][u][t]))
                avg_vars_per_user_ts.append(np.mean(vars_vids[v][u][t]))
                avg_errs_per_user_ts.append(np.mean(err_vids[v][u][t]))
                for elem in range(len(vars_vids[v][u][t])):
                    vars_ts[v][u].append(vars_vids[v][u][t][elem])
                    errs_ts[v][u].append(err_vids[v][u][t][elem])
                    errs_vid_ts[v].append(err_vids[v][u][t][elem])
                    vars_vid_ts[v].append(vars_vids[v][u][t][elem])
                    err_vid[v].append(err_vids[v][u][t][elem])
                    var_vid[v].append(vars_vids[v][u][t][elem])
        c = 1
        for thresh in thresholds:
            greq_leq[v][thresh] = []
            print("VIDEO", v, "\n",thresh)
            for elem in var_vid[v]:
                if elem > thresh:
                    #print(elem)
                    greq_leq[v][thresh].append(1)
                    sig_var = True
                else:
                    greq_leq[v][thresh].append(0)
                    sig_var = False
                c += 1
##                print(len(greq_leq[v][thresh]), len(err_vid[v]), len(var_vid[v]))
            vid_gl_counts[v].append(greq_leq[v][thresh].count(1))
            print(greq_leq[v][thresh].count(1))
            if greq_leq[v][thresh].count(0) != len(greq_leq[v][thresh]):
                #print("Video",v, "user",u,"AUROC",metrics.roc_auc_score(greq_leq[v][u], errs_ts[v][u]))
                AUROCS[v].append(metrics.roc_auc_score(greq_leq[v][thresh], err_vid[v]))
##                plt.style.use('fivethirtyeight')
##                plt.rcParams.update({'font.size': 7})
####                plt.scatter(errs_ts[v][u],vars_ts[v][u],marker='.',  color='b')
####                print("vid",v, "User", u, "p",pcorr, "s",scorr)
##                plt.scatter(avg_errs_per_user_ts,avg_vars_per_user_ts,marker='.',  color='b')
##                plt.xlabel("Average Orthodromic Distance")
##                plt.ylabel("Average Empirical Variance")
##                plt.title("%s, User %s"%(v,u))
##                plt.show()
                #plt.grid()
##                plt.savefig(results_folder+'/%s_user_%s'%(v,u)+'.pdf')
                plt.clf()            
        pcorr, _ = pearsonr(avg_errs_per_ts, avg_vars_per_ts)
        scorr, _ = spearmanr(avg_errs_per_ts, avg_vars_per_ts)
        print("vid",v, "p",pcorr, "s",scorr)
##        plt.style.use('fivethirtyeight')
##        plt.rcParams.update({'font.size': 7})
        plt.grid()
        plt.scatter(avg_errs_per_ts,avg_vars_per_ts,marker='.',  color='b')
        z= np.polyfit(avg_errs_per_ts,avg_vars_per_ts,1)
##        p = np.poly1d(z)
        #plt.plot(avg_errs_per_ts,p(avg_errs_per_ts),"r--", linewidth=1)
        
        plt.xlabel("Average Orthodromic Distance")
        plt.ylabel("Average Empirical Variance")
        plt.title("%s"%(v))
        #plt.grid()
        plt.show()
        plt.clf()
    print([AUROCS[k] for k in AUROCS.keys()])
    [print("Video %s,  Counts over thres: %s, Thresholds %s \n" %(vid, vid_gl_counts[vid], thresholds)) for vid in vid_gl_counts.keys()]
    [plt.plot(thresholds, AUROCS[k]) for k in AUROCS.keys()]

    normalized_gl_counts = {}
    for vid in vid_gl_counts.keys():
        normalized_gl_counts[vid] = []
        for elem in vid_gl_counts[vid]:
            normalized_gl_counts[vid].append(elem/sum(vid_gl_counts[vid]))
    [plt.plot(thresholds, normalized_gl_counts[vid], alpha=0.3, label=vid) for vid in vid_gl_counts.keys()]
    plt.xscale("log")
    #plt.yscale("log")
##    plt.ylabel("Count of variances over threshold")
    plt.ylabel("AUROC")
    plt.xlabel('Threshold on variance (aleatoric uncertainty estimate)')
    plt.legend([k for k in AUROCS.keys()])
##    plt.ylabel('AUROC')
    plt.show()
    
def check_for_orthdist_outliers_align(results_folder):
    var_vids = results_folder + '/var_per_vid.pkl'
    err_vids = results_folder + '/err_per_vid_user.pkl'
    with open(var_vids, 'rb') as f:
        vars_vids = pickle.load(f)

    with open(err_vids, 'rb') as f:
        err_vid_users = pickle.load(f)

    interval = range(25)
    videos = []
    avg_vars_per_vid = []
    avg_errs_per_vid = []
    greq_leq = {}
    errs_ts = {}
    vars_ts = {}
    AUROCs = {}
    thresholds = [0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,
                  0.6,0.7,0.8,0.9,1,1.1,1.2]
    # pos_only_vars[video][timestep][vars_for_timestep]
    err_vids = {}
    for k in vars_vids.keys():
        print(k)
        greq_leq[k] = []
        videos.append(k)
        avg_vars_per_ts = []
        avg_err_per_ts = []
        errs_ts[k] = []
        err_vids[k] = {}
        vars_ts[k] = []
        AUROCs[k] = []
        p = 0
        for u in err_vid_users[k].keys():
            for t in interval:
                err_vids[k][t] = []
                err_vids[k][t].append(err_vid_users[k][u][t])
        for thresh in thresholds:
            print("progress", p, "/", len(thresholds))
            p += 1
            for t in interval:
                avg_vars_per_ts.append(np.mean(vars_vids[k][t][0]))
                avg_err_per_ts.append(np.mean(err_vids[k][t][0]))
            #thresh = 1.3
                i = 0
                for elem in err_vids[k][t][0]:
                    if elem > thresh:
                        greq_leq[k].append(1)
                    else:
                        greq_leq[k].append(0)
                    #errs_ts[k].append(err_vids[k][t][i])
                    vars_ts[k].append(vars_vids[k][t][i])
                    i += 1
            AUROCs[k].append(metrics.roc_auc_score(greq_leq[k], vars_ts[k]))
                #print("Video",k,"AUROC",metrics.roc_auc_score(greq_leq[k], vars_ts[k]))

    [plt.plot(thresholds, AUROCs[k]) for k in AUROCs.keys()]
    plt.legend([k for k in AUROCs.keys()])
    plt.xlabel('Threshold on orthodromic distance')
    plt.ylabel('AUROC')
    plt.show()

if scatter_plots_ts:
    get_and_compile_per_step_corr(results_folder)

check_for_var_user_outliers(results_folder)
#check_for_var_outliers_align(results_folder)
##check_for_orthdist_outliers_align(results_folder)

pcorr, scorr, avg_vars, avg_errs = get_and_compile_global_vars_corr(results_folder)
print("pearson corr: ",pcorr, "\nSpearman corr:",scorr)
##plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 7})

##fig = plt.figure()
##ax = plt.axes(projection='3d')
##ax.plot3D(range(25), avg_vars, avg_errs, 'red')
##ax.set_xlabel('Prediction step')
##ax.set_ylabel('Average predictive variance')
##ax.set_zlabel('Average orthodromic distance')
plt.plot(avg_errs, avg_vars, 'red')
for tstep in range(25):
    plt.annotate('%s'%(tstep+1), xy=(avg_errs[tstep], avg_vars[tstep]),textcoords='data')
plt.xlabel('Average orthodromic distance')
plt.ylabel('Average predictive variance (estimated aleatoric uncertainty)')
plt.grid()
plt.show()
