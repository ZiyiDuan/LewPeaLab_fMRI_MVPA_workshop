# -*- coding: utf-8 -*-

"""
@author: Zoe Duan
@file: #clearmen_localizer_classifier_subcategory.py
@time: 11/6/21 2:59 下午
@desp: Build a classifier for functional localizer task data to distinguish 9 sub-categories
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
from utils import load_data, load_labels, label2TR, shift_timing, reshape_data, blockwise_sampling, decode, normalize

'''
define some useful functions
'''

def subsample(labels, num):
    '''
    subsample rest states TRs and return the indexes
    :param labels: labels for different TRs, 0=rest, 1=stimulus presented
    :param num: 1 = subsample the rest TRs to the same as stimulus presented TRs, 0 = delete all rest TRs
    :return: indexes of chosen TRs
    '''

    rest_indexes = np.squeeze(np.where(labels==0))
    stimulus_indexes = np.squeeze(np.where(labels==1))

    stimulus_len = len(stimulus_indexes)

    if num == 0:
        subsampled_indexes = stimulus_indexes
    else:
        rest_indexes_subsampled = np.random.choice(rest_indexes, stimulus_len, replace=False)
        subsampled_indexes = np.sort(np.append(rest_indexes_subsampled, stimulus_indexes))
    return subsampled_indexes

'''
Set experimental parameters
'''

TR = 0.46
shift_size = 10
n_runs_localizer = 5

cwd = os.getcwd()
datadir = cwd + '/BIDS/'

'''
load localizer design matrix
'''
design_matrix = pd.read_csv(datadir + 'localizer_design_matrix_new.csv')
# The new design matrix changed the category and subcategory label into 0 when its rest states

# get category and subcategory labels and stimulus labels
run_ids = np.array(design_matrix['run'])
category_label_raw = np.array(design_matrix['category']).reshape(len(design_matrix['category']), 1)
subcategory_label_raw = np.array(design_matrix['subcategory']).reshape(len(design_matrix['subcategory']), 1)
stimulus_raw = np.array(design_matrix['stimulus']).reshape(len(design_matrix['stimulus']), 1)
# 0=rest, 1=stimulus presented

# Shift the data some amount
category_label_shifted = shift_timing(category_label_raw, shift_size)
subcategory_label_shifted = shift_timing(subcategory_label_raw, shift_size)
stimulus_raw_shifted = shift_timing(stimulus_raw, shift_size)

# change design matrix
design_matrix['category'] = category_label_shifted
design_matrix['subcategory'] = subcategory_label_shifted
design_matrix['stimulus'] = stimulus_raw_shifted


# subsample data to balance the number of rest TRs and stimulus TRs
stimulus_raw = design_matrix['stimulus']

# delete all rest TRs only remain stimulus datapoints
norest_ids = subsample(stimulus_raw, 0)
design_matrix_norest = design_matrix.loc[norest_ids, :]
# get other information
run_ids_norest = np.array(design_matrix_norest['run'])
block_ids_norest = np.array(design_matrix_norest['block'])
category_label_norest = np.array(design_matrix_norest['category'])
pre_subcategory_label_norest = np.array(design_matrix_norest['subcategory'])
# change subcategory labels into 1~9
subcategory_label_norest = np.zeros(pre_subcategory_label_norest.shape)
for i in range(len(design_matrix_norest)):
    category_label_i = category_label_norest[i]
    pre_subcategory_label_i = pre_subcategory_label_norest[i]
    subcategory_label_norest[i] = (category_label_i - 1)*3 +  pre_subcategory_label_i




'''
load the data
'''

# data_format = 'wholebrain'
data_format = 'masked'

sub_ids = [27, 34, 44, 77]

# pre-allocate list to store image data
img_data_all = {}

for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    if data_format == 'wholebrain':
        img_data_id = np.load(datadir + sub +'_task-localizer_allruns_raw.npy')
    elif data_format == 'masked':
        img_data_id = np.load(datadir + sub + '_task-localizer_allruns_masked.npy')
    img_data_all[sub] = img_data_id
    print('You have loaded {} data from {}'.format(data_format, sub))


'''
subCategory classifier
'''

# classify nine sub-categories
categories = ['actor', 'musician', 'politician', 'apple', 'grape', 'pear', 'beach', 'bridge', 'mountain']

results = {}

for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    img_data = img_data_all[sub]
    # normalize raw data within each run
    img_data_normalized = normalize(img_data, run_ids=run_ids, n_runs=n_runs_localizer)
    # select no-rest data
    img_normalized_norest = img_data_normalized[norest_ids, :]
    
    print('You have preprocessed data for {}, now begin the classification'.format(sub))


    # Nested cross-validation: Hyper-parameter selection
    sp = PredefinedSplit(run_ids_norest)

    # Print out the training, validation, and testing set (the indexes that belong to each group)
    # Outer loop:
    # Split training (including validation) and testing set
    sp = PredefinedSplit(run_ids_norest)
    for outer_idx, (train, test) in enumerate(sp.split()):
        train_run_ids = run_ids_norest[train]
        print('Outer loop % d:' % outer_idx)
        print('Testing: ')
        print(test)

        # Inner loop (implicit, in GridSearchCV):
        # split training and validation set
        sp_train = PredefinedSplit(train_run_ids)
        for inner_idx, (train_inner, val) in enumerate(sp_train.split()):
            print('Inner loop %d:' % inner_idx)
            print('Training: ')
            print(train[train_inner])
            print('Validation: ')
            print(train[val])

    # pre-allocate space to store results for each cross-validation
    C_best_clf_1 = np.array([])
    y_true = np.array([])
    y_pred_clf_1 = np.array([])
    n_features_clf_1 = np.array([])
    n_features_probability_clf_1 = np.array([])
    
    # Outer loop:
    # Split training (including validation) and testing set
    for train, test in tqdm(sp.split()):
        # Pull out the sample data
        train_run_ids = run_ids_norest[train]
        train_data = img_normalized_norest[train, :]
        test_data = img_normalized_norest[test, :]
        train_label = subcategory_label_norest[train]
        test_label = subcategory_label_norest[test]
        y_true = np.hstack((y_true, test_label))

        # Inner loop (implicit, in GridSearchCV):
        # Split training and validation set
        sp_train = PredefinedSplit(train_run_ids)

        # Search over different regularization parameters: smaller values specify stronger regularization.
        parameters = {'C': [10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3]}
        inner_clf = GridSearchCV(
            estimator=LogisticRegression(penalty='l2'),
            param_grid=parameters,
            cv=sp_train,
            return_train_score=True)
        inner_clf_1 = inner_clf

        # Classification 1: do voxel selection by using selectFpr
        selected_voxels_clf_1 = SelectFpr(f_classif, alpha=0.05).fit(train_data, train_label)
        n_features_clf_1_i = sum(selected_voxels_clf_1.pvalues_ < 0.05)
        n_features_clf_1 = np.append(n_features_clf_1, n_features_clf_1_i)
        n_features_probability_clf_1_i = n_features_clf_1_i/len(selected_voxels_clf_1.pvalues_)
        n_features_probability_clf_1 = np.append(n_features_probability_clf_1, n_features_probability_clf_1_i)
        print('You have selected {} features based on Fpr for {}, the probability is {}'.format(n_features_clf_1_i, sub, n_features_probability_clf_1_i))


        # select the best hyperparameters for clf 1
        inner_clf_1.fit(selected_voxels_clf_1.transform(train_data), train_label)
        C_best_clf_1_i = inner_clf_1.best_params_['C']
        C_best_clf_1 = np.append(C_best_clf_1, C_best_clf_1_i)
        print('The best C in current cv for clf 1 is', C_best_clf_1_i)

        # Train the classifier with the best hyperparameter using training and validation set
        classifier_1 = LogisticRegression(penalty='l2', C=C_best_clf_1_i)
        clf_1 = classifier_1.fit(selected_voxels_clf_1.transform(train_data), train_label)

        # Test the classifier
        print('The correct labels in current cv is', test_label)

        pred_clf_1 = clf_1.predict(selected_voxels_clf_1.transform(test_data))
        y_pred_clf_1 = np.hstack((y_pred_clf_1, pred_clf_1))
        print('Your predictions in current cv for clf 1 is', pred_clf_1)

    # save results for each subject
    results[sub] = {}
    results[sub]['clf_1'] = {}
    results[sub]['clf_1']['y_true'] = y_true

    results[sub]['clf_1']['C_best'] = C_best_clf_1
    results[sub]['clf_1']['y_pred'] = y_pred_clf_1
    results[sub]['clf_1']['n_features'] = n_features_clf_1
    results[sub]['clf_1']['n_features_probability'] = n_features_probability_clf_1

    # get the confusion matrix for clf 1
    confusion_matrix_clf_1 = confusion_matrix(y_true, y_pred_clf_1)
    # transform it into probability
    confusion_matrix_clf_1 = confusion_matrix_clf_1 / len(y_true) * 3
    results[sub]['clf_1']['confusion matrix'] = confusion_matrix_clf_1
    print('Confusion matrix for {} for clf 1 is {}'.format(sub, confusion_matrix_clf_1))


    # save sub-level results to the disk
    f = open('classification_results_localizer_{}_{}'.format(data_format, sub), "wb")
    pickle.dump(results[sub], f)
    f.close


    # plot the confusion matrix
    fig_title = '{} confusion matrix for {} with {}'.format(data_format, sub, 'clf_1')
    fig_name = '{}_confusion_matrix_{}_{}.png'.format(data_format, 'clf_1', sub)

    f, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.set_title(fig_title, fontsize=30)
    sns.heatmap(confusion_matrix_clf_1, vmin=0, vmax=0.4, cmap='YlGnBu', linewidths=0.2,
                annot=True, annot_kws={"fontsize": 15}, cbar=True)
    ax.set_xticklabels(labels=categories, fontsize=20, rotation=90)
    ax.set_yticklabels(labels=categories, fontsize=20, rotation=0)
    ax.set_xlabel('Predicted category', fontsize=30)
    ax.set_ylabel('Actual category', fontsize=30)
    # ax.xaxis.set_label_position('top')
    # ax.xaxis.tick_top()
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


# save all subjects data into disk
f = open('classification_results_localizer_{}_all'.format(data_format), "wb")
pickle.dump(results, f)
f.close

# combine all subjects' results and plot the confusion matrix
# get all confusion matrix
clf_ind = 'clf_1'
confusion_matrix_all = {}
confusion_matrix_mean = np.array([])
for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    confusion_matrix_all[sub] = results[sub][clf_ind]['confusion matrix']
    if len(confusion_matrix_mean) == 0:
        confusion_matrix_mean = confusion_matrix_all[sub]
    else:
        confusion_matrix_mean = confusion_matrix_mean + confusion_matrix_all[sub]

confusion_matrix_mean = confusion_matrix_mean/len(sub_ids)

# plot group average results
fig_title = '{} confusion matrix for {} subjects'.format(data_format, len(sub_ids))
fig_name = '{}_confusion_matrix_{}_{}_subjects.png'.format(data_format, clf_ind, len(sub_ids))

f, ax = plt.subplots(1, 1, figsize=(15, 12))
ax.set_title(fig_title, fontsize=30, )
sns.heatmap(confusion_matrix_mean, vmin=0, vmax=0.4, cmap='YlGnBu', linewidths=0.2,
            annot=True, annot_kws={"fontsize":15}, cbar=True)
ax.set_xticklabels(labels=categories, fontsize=20, rotation=90)
ax.set_yticklabels(labels=categories, fontsize=20, rotation=0)
ax.set_xlabel('Predicted category', fontsize=30)
ax.set_ylabel('Actual category', fontsize=30)
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()
plt.savefig(fig_name, bbox_inches='tight')
plt.show()


