# -*- coding: utf-8 -*-

"""
@author: Zoe Duan
@file: #clearmen_study_classifier_test.py
@time: 11/8/21 3:34 下午
@desp: test the classifier for study task
@data:
    preprocessed four subjects' data, organized in BIDS
    TR = 460ms
Functional localizer task:
    5 runs, 3 categories with 3 subcategories for each, 90 trials each category, 270 trials in total
    Each run: 54 trials, 18 blocks, 3 trials each block, stimuli in the same block has the same category and subcategory
    Each trial: 3 TRs, 5~10 TRs between trials
Central study task:
    6 runs, 5 cognitive operations: maintain, replace subcategory, replace category, suppress, clear
    Each run: 60 trials, 12 trials for each operation, 40 TRs before each run
    Each trial: 6 TRs for presenting image, 6 TRs for cognitive operation, 5~9 TRs inter-trial fixation
"""


import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
# Import fMRI and general analysis libraries
import nibabel as nib
import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
import os
import pickle

from tqdm import trange, tqdm

# Import plotting library
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib notebook

# Import machine learning libraries
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest, SelectFpr
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import sem
from copy import deepcopy
from sklearn.metrics import confusion_matrix


# We still have to import the functions of interest
from utils import shift_timing, normalize


'''
Set experimental parameters
'''

TR = 0.46
shift_size = 10
n_runs = 6
n_trials = 60

cwd = os.getcwd()
datadir = cwd + '/BIDS/'

'''
load study design matrix
'''
design_matrix = pd.read_csv(datadir + 'study_volume_design_matrix.csv')

TR_length = len(design_matrix)

# get labels
run_labels = np.array(list(design_matrix['run']))
trial_labels = np.array(list(design_matrix['trial']))
operation_labels = np.array(list(design_matrix['condition']))   # delete sub-category replace, condition=3
category_labels = np.array(list(design_matrix['category']))
new_category_labels = np.array(list(design_matrix['new_category']))
presentation_labels = np.array(list(design_matrix['presentation']))    # 1=stimuli present, 2=operation, 0=fixation
manipulation_labels = np.array(list(design_matrix['manipulation']))     # 1=operation periods we considered, including the operation and fixation periods

# define category and new_category labels that change 0 into corespondent labels to draw the timecourse
all_category_labels = np.zeros(TR_length, dtype=int)
all_new_category_labels = np.zeros(TR_length, dtype=int)

for r in range(n_runs):
    curr_trial_ind = np.where(run_labels == r)[0]
    for i in range(1, n_trials+1):
        ind = np.where(trial_labels[curr_trial_ind]==i)[0]
        trial_ind = curr_trial_ind[ind]
        category_ind = category_labels[trial_ind]
        category_label_ind = max(category_ind)
        all_category_labels_ind = np.repeat(category_label_ind, len(trial_ind))
        all_category_labels[trial_ind] = all_category_labels_ind

        new_category_ind = new_category_labels[trial_ind]
        new_category_label_ind = max(new_category_ind)
        all_new_category_labels_ind = np.repeat(new_category_label_ind, len(trial_ind))
        all_new_category_labels[trial_ind] = all_new_category_labels_ind

all_category_labels = all_category_labels.reshape(TR_length, 1)
all_new_category_labels = all_new_category_labels.reshape(TR_length, 1)


# Shift the data some amount
run_labels_shifted = np.squeeze(shift_timing(run_labels.reshape(TR_length, 1), shift_size)).astype(int)
trial_labels_shifted = np.squeeze(shift_timing(trial_labels.reshape(TR_length, 1), shift_size)).astype(int)
operation_labels_shifted = np.squeeze(shift_timing(operation_labels.reshape(TR_length, 1), shift_size)).astype(int)
category_labels_shifted = np.squeeze(shift_timing(all_category_labels.reshape(TR_length, 1), shift_size)).astype(int)
new_category_labels_shifted = np.squeeze(shift_timing(all_new_category_labels.reshape(TR_length, 1), shift_size)).astype(int)
presentation_labels_shifted = np.squeeze(shift_timing(presentation_labels.reshape(TR_length, 1), shift_size)).astype(int)
manipulation_labels_shifted = np.squeeze(shift_timing(manipulation_labels.reshape(TR_length, 1), shift_size)).astype(int)



'''
load trained classifier for all subjects
'''

f = open('study_trained_classifier_masked_all', 'rb')
clf_trained_all = pickle.load(f)
f.close


'''
load masked data for study phase
'''
# data_format = 'wholebrain'
data_format = 'masked'

# sub_ids = [27]
sub_ids = [27, 34, 44, 77]

# pre-allocate list to store image data
img_data_all = {}

for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    if data_format == 'wholebrain':
        img_data_id = np.load(datadir + sub +'_task-study_allruns_raw.npy')
    elif data_format == 'masked':
        img_data_id = np.load(datadir + sub + '_task-study_allruns_masked.npy')
    img_data_all[sub] = img_data_id
    print('You have loaded {} study phase data from {}'.format(data_format, sub))


'''
get category classifier evidence
'''
categories = ['face', 'fruit', 'scene']
operations = ['maintain', 'replace', 'suppress', 'clear']

# define the total number of TR across the timecourses
TR_trial = 30
# define the total number of TR as baseline correction
TR_baseline = 6

results = {}

for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)

    results[sub] = {}

    img_data = img_data_all[sub]

    # Normalize raw data within each run
    img_data_normalized = normalize(img_data, run_labels, n_runs=n_runs)

    print('You have preprocessed data for {}, now begin the classification'.format(sub))

    # find the trained classifier for current sub
    selected_features = clf_trained_all[sub]['clf_1']['selected_features']
    clf_trained = clf_trained_all[sub]['clf_1']['clf_trained']
    # test the study data and get classifier evidence for all categories
    clf_evidence = clf_trained.predict_proba(selected_features.transform(img_data_normalized))
    results[sub]['clf_evidence'] = clf_evidence

    print('You have got the classifier evidence for {}'.format(sub))


    maintain_evidence = np.array([])
    replace_old_evidence = np.array([])
    replace_new_evidence = np.array([])
    suppress_evidence = np.array([])
    clear_evidence = np.array([])
    baseline_evidence = np.array([])

    for r in range(n_runs):
        curr_trial_ind = np.where(run_labels == r)[0]
        for t in range(1, n_trials+1):
            ind = np.where(trial_labels[curr_trial_ind] == t)[0]
            trial_ind = curr_trial_ind[ind]
            onset_ind = trial_ind[0]


            # define trial-wise evidence for different conditions
            trial_maintain_evidence = np.array([])
            trial_replace_old_evidence = np.array([])
            trial_replace_new_evidence = np.array([])
            trial_suppress_evidence = np.array([])
            trial_clear_evidence = np.array([])
            trial_baseline_evidence = np.array([])

            curr_category_label = int(all_category_labels[onset_ind])
            curr_new_category_label = int(all_new_category_labels[onset_ind])
            curr_operation = int(operation_labels[onset_ind])

            if curr_operation == 1:
                trial_maintain_evidence = clf_evidence[onset_ind : onset_ind + TR_trial, curr_category_label - 1]
            elif curr_operation == 2:
                trial_replace_old_evidence = clf_evidence[onset_ind : onset_ind + TR_trial, curr_category_label - 1]
                trial_replace_new_evidence = clf_evidence[onset_ind : onset_ind + TR_trial, curr_new_category_label - 1]
            elif curr_operation == 4:
                trial_suppress_evidence = clf_evidence[onset_ind : onset_ind + TR_trial, curr_category_label - 1]
                trial_baseline_evidence = np.delete(clf_evidence[onset_ind : onset_ind + TR_trial, ],
                                                    curr_category_label - 1, 1)
                # random choose irrelevant items
                idx = np.random.choice([0, 1])
                trial_baseline_evidence = trial_baseline_evidence[:, idx]
            elif curr_operation == 5:
                trial_clear_evidence = clf_evidence[onset_ind : onset_ind + TR_trial, curr_category_label - 1]
                trial_baseline_evidence = np.delete(clf_evidence[onset_ind: onset_ind + TR_trial, ],
                                                    curr_category_label - 1, 1)
                # random choose irrelevant items
                idx = np.random.choice([0, 1])
                trial_baseline_evidence = trial_baseline_evidence[:, idx]

            # concat trial-wise evidence vertically for different conditions
            if len(trial_maintain_evidence) != 0:
                if len(maintain_evidence) == 0:
                    maintain_evidence = trial_maintain_evidence
                else:
                    maintain_evidence = np.vstack((maintain_evidence, trial_maintain_evidence))
            if len(trial_replace_old_evidence) != 0:
                if len(replace_old_evidence) == 0:
                    replace_old_evidence = trial_replace_old_evidence
                else:
                    replace_old_evidence = np.vstack((replace_old_evidence, trial_replace_old_evidence))
            if len(trial_replace_new_evidence) != 0:
                if len(replace_new_evidence) == 0:
                    replace_new_evidence = trial_replace_new_evidence
                else:
                    replace_new_evidence = np.vstack((replace_new_evidence, trial_replace_new_evidence))
            if len(trial_suppress_evidence) != 0:
                if len(suppress_evidence) == 0:
                    suppress_evidence = trial_suppress_evidence
                else:
                    suppress_evidence = np.vstack((suppress_evidence, trial_suppress_evidence))
            if len(trial_clear_evidence) != 0:
                if len(clear_evidence) == 0:
                    clear_evidence = trial_clear_evidence
                else:
                    clear_evidence = np.vstack((clear_evidence, trial_clear_evidence))
            if len(trial_baseline_evidence) != 0:
                if len(baseline_evidence) == 0:
                    baseline_evidence = trial_baseline_evidence
                else:
                    baseline_evidence = np.vstack((baseline_evidence, trial_baseline_evidence))

    # average evidence across trials
    maintain_evidence_mean = np.mean(maintain_evidence, 0)
    replace_old_evidence_mean = np.mean(replace_old_evidence, 0)
    replace_new_evidence_mean = np.mean(replace_new_evidence, 0)
    suppress_evidence_mean = np.mean(suppress_evidence, 0)
    clear_evidence_mean = np.mean(clear_evidence, 0)
    baseline_evidence_mean = np.mean(baseline_evidence, 0)

    # baseline correct averaged evidence
    maintain_evidence_baseline = np.mean(maintain_evidence_mean[0:TR_baseline])
    maintain_evidence_final = maintain_evidence_mean - maintain_evidence_baseline
    replace_old_evidence_baseline = np.mean(replace_old_evidence_mean[0:TR_baseline])
    replace_old_evidence_final = replace_old_evidence_mean - replace_old_evidence_baseline
    replace_new_evidence_baseline = np.mean(replace_new_evidence_mean[0:TR_baseline])
    replace_new_evidence_final = replace_new_evidence_mean - replace_new_evidence_baseline
    suppress_evidence_baseline = np.mean(suppress_evidence_mean[0:TR_baseline])
    suppress_evidence_final = suppress_evidence_mean - suppress_evidence_baseline
    clear_evidence_baseline = np.mean(clear_evidence_mean[0:TR_baseline])
    clear_evidence_final = clear_evidence_mean - clear_evidence_baseline
    baseline_evidence_baseline = np.mean(baseline_evidence_mean[0:TR_baseline])
    baseline_evidence_final = baseline_evidence_mean - baseline_evidence_baseline


    # plot time course of classifier evidence for different conditions
    data_WM = {}
    data_WM['sub'] = sub
    data_WM['maintain'] = maintain_evidence_final
    data_WM['replace_old'] = replace_old_evidence_final
    data_WM['replace_new'] = replace_new_evidence_final
    data_WM['baseline'] = baseline_evidence_final
    df_WM = pd.DataFrame(data_WM)

    # save results
    results[sub]['timecourses_WM'] = df_WM

    # plot results
    f, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title('Timecourses for neural decoding of a WM item for {}'.format(sub))
    sns.lineplot(data=df_WM)
    plt.savefig('timecourses_decoding_WM_{}.png'.format(sub))
    plt.show()



    # plot trajectory for removal of an item from WM
    data_removal = {}
    data_removal['sub'] = sub
    data_removal['replace'] = replace_old_evidence_final - maintain_evidence_final
    data_removal['suppress'] = suppress_evidence_final - maintain_evidence_final
    data_removal['clear'] = clear_evidence_final - maintain_evidence_final
    df_removal = pd.DataFrame(data_removal)

    # save results
    results[sub]['timecourses_removal'] = df_removal

    # plot results
    f, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title('Trajectory for removal of an item from WM for {}'.format(sub))
    sns.lineplot(data=df_removal)
    ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1])
    plt.savefig('timecourses_decoding_removal_{}.png'.format(sub))
    plt.show()


# plot group-level timecourses
df_WM_all = pd.DataFrame(columns=['sub', 'maintain', 'replace_old', 'replace_new', 'baseline'])
for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    df_WM_all = pd.concat((df_WM_all, results[sub]['timecourses_WM']))

f, ax = plt.subplots(1, 1, figsize=(12, 8))
sns.set_style("ticks")
ax.set_title('Timecourses for neural decoding of a WM item for {} subjects'.format(len(sub_ids)), fontsize=20)
sns.lineplot(data=df_WM_all, ci=95)
sns.despine()
ax.axhline(0, ls='--', c='black')
ax.set_xlabel('TR', loc='right', fontsize=20)
ax.set_ylabel('Category classifier evidence', fontsize=20)
plt.savefig('timecourses_decoding_WM_all.png')
plt.show()


df_removal_all = pd.DataFrame(columns=['sub', 'replace', 'suppress', 'clear'])
for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    df_removal_all = pd.concat((df_removal_all, results[sub]['timecourses_removal']))

f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_title('Trajectory for removal of an item from WM for {} subjects'.format(len(sub_ids)))
sns.lineplot(data=df_removal_all, ci=95)
sns.despine()
ax.axhline(0, ls='--', c='black')
ax.set_xlabel('TR', loc='right', fontsize=20)
ax.set_ylabel('classifier evidence\n(removal - maintain)', fontsize=20)
ax.set_yticks([-0.1,-0.05,0,0.05,0.1])
plt.savefig('timecourses_decoding_removal_all.png')
plt.show()




