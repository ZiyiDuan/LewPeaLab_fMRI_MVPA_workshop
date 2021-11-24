# -*- coding: utf-8 -*-

"""
@author: Zoe Duan
@file: #clearmen_study_classifer_operation.py
@time: 11/11/21 10:15 上午
@desp: build a classifier for study task to distinguish 4 operations
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
define some useful functions
'''



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
operation_labels = np.array(list(design_matrix['condition']))
# delete sub-category replace, condition=3; 1=maintain, 2=replace category-level, 4=suppress, 5=clear
presentation_labels = np.array(list(design_matrix['presentation']))    # 1=stimuli present, 2=operation, 0=fixation
manipulation_labels = np.array(list(design_matrix['manipulation']))     # 1=operation periods we considered, including the operation and fixation periods


# Shift the data some amount
run_labels_shifted = np.squeeze(shift_timing(run_labels.reshape(TR_length, 1), shift_size)).astype(int)
trial_labels_shifted = np.squeeze(shift_timing(trial_labels.reshape(TR_length, 1), shift_size)).astype(int)
operation_labels_shifted = np.squeeze(shift_timing(operation_labels.reshape(TR_length, 1), shift_size)).astype(int)
presentation_labels_shifted = np.squeeze(shift_timing(presentation_labels.reshape(TR_length, 1), shift_size)).astype(int)
manipulation_labels_shifted = np.squeeze(shift_timing(manipulation_labels.reshape(TR_length, 1), shift_size)).astype(int)



'''
load whole brain data for study phase
'''
data_format = 'wholebrain'
# data_format = 'masked'

sub_ids = [44]
# sub_ids = [27, 34, 44, 77]

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
classify different mental operations
'''
operations = ['maintain', 'replace', 'suppress', 'clear']
# operations = ['maintain', 'replace_category', 'replace_subcategory', 'suppress', 'clear']

results = {}

for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    img_data = img_data_all[sub]

    # select operation periods, and delete replace-subcategory, operation=3
    operation_inds = np.where((manipulation_labels_shifted == 1)&(operation_labels_shifted!=3) & (operation_labels_shifted!=0))[0]
    img_data_operation = img_data[operation_inds, :]
    run_labels_operation = run_labels_shifted[operation_inds]
    operation_labels_operation = operation_labels_shifted[operation_inds]


    # Normalize raw data within each run
    img_normalized_operation = normalize(img_data_operation, run_labels_operation, n_runs=n_runs)

    print('You have preprocessed data for {}, now begin the classification'.format(sub))


    # Nested cross-validation: Hyper-parameter selection
    sp = PredefinedSplit(run_labels_operation)

    # Print out the training, validation, and testing set (the indexes that belong to each group)
    # Outer loop:
    # Split training (including validation) and testing set
    sp = PredefinedSplit(run_labels_operation)
    for outer_idx, (train, test) in enumerate(sp.split()):
        train_run_ids = run_labels_operation[train]
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
        train_run_ids = run_labels_operation[train]
        train_data = img_data_operation[train, :]
        test_data = img_data_operation[test, :]
        train_label = operation_labels_operation[train]
        test_label = operation_labels_operation[test]
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
        # search the best parameters for three different classifiers
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
    confusion_matrix_clf_1 = confusion_matrix_clf_1/len(y_true)*3
    results[sub]['clf_1']['confusion matrix'] = confusion_matrix_clf_1
    print('Confusion matrix for {} for clf 1 is {}'.format(sub, confusion_matrix_clf_1))
    # plot the confusion matrix
    if data_format == 'wholebrain':
        fig_title = 'wholebrain operation confusion matrix for {} with {}'.format(sub, 'clf_1')
        fig_name = 'wholebrain_operation_confusion_matrix_{}_{}.png'.format('clf_1', sub)
    elif data_format == 'masked':
        fig_title = 'masked operation confusion matrix for {} with {}'.format(sub, 'clf_1')
        fig_name = 'maked_operation_confusion_matrix_{}_{}.png'.format('clf_1', sub)

    f, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title(fig_title)
    sns.heatmap(confusion_matrix_clf_1, vmin=0, vmax=1, cmap='YlGnBu', annot=True)
    ax.set_xticklabels(operations)
    ax.set_yticklabels(operations)
    plt.savefig(fig_name)
    plt.show()

    # save sub-level results to the disk
    if data_format == 'wholebrain':
        f = open('classification_results_study_operation_wholebrain_{}'.format(sub), "wb")
    elif data_format == 'masked':
        f = open('classification_results_study_operation_masked_{}'.format(sub), "wb")
    pickle.dump(results[sub], f)
    f.close


# save all subjects data into disk
if data_format == 'wholebrain':
    f = open('classification_results_study_operation_wholebrain_all', "wb")
elif data_format == 'masked':
    f = open('classification_results_study_operation_masked_all', "wb")
pickle.dump(results, f)
f.close

