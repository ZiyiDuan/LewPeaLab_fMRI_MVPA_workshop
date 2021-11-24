# -*- coding: utf-8 -*-

"""
@author: Zoe Duan
@file: #clearmen_study_classifier_training.py
@time: 11/7/21 3:31 下午
@desp: training the classifier for study task based on all data in localizer task
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
run_labels = np.array(design_matrix['run'])
category_label_raw = np.array(design_matrix['category']).reshape(len(design_matrix['category']), 1)
subcategory_label_raw = np.array(design_matrix['subcategory']).reshape(len(design_matrix['subcategory']), 1)
stimulus_raw = np.array(design_matrix['stimulus']).reshape(len(design_matrix['stimulus']), 1)
# 0=rest, 1=stimulus presented

# Shift the data some amount
category_label_shifted = shift_timing(category_label_raw, shift_size)
subcategory_label_shifted = shift_timing(subcategory_label_raw, shift_size)
stimulus_shifted = shift_timing(stimulus_raw, shift_size)

# change design matrix
design_matrix['category'] = category_label_shifted
design_matrix['subcategory'] = subcategory_label_shifted
design_matrix['stimulus'] = stimulus_shifted


# delete all rest TRs only remain stimulus datapoints
norest_ids = subsample(stimulus_shifted, 0)
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
category classifier
'''

# classify three categories
categories = ['face', 'fruit', 'scene']

results = {}


for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    img_data = img_data_all[sub]
    # normalize raw data within each run
    img_data_normalized = normalize(img_data, run_ids=run_labels, n_runs=n_runs_localizer)
    # select no-rest data
    img_normalized_norest = img_data_normalized[norest_ids, :]

    print('You have preprocessed data for {}, now begin the classification'.format(sub))

    # pre-allocate space to store results for each cross-validation
    C_best_clf_1 = []
    selected_features_clf_1 = []
    n_features_clf_1 = []
    trained_clf_1 = []

    # gridsearch for best parameters based on all data
    x = img_normalized_norest
    y = category_label_norest
    # Split training and validation set
    sp = PredefinedSplit(run_ids_norest)

    # Search over different regularization parameters: smaller values specify stronger regularization.
    parameters = {'C': [10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3]}
    inner_clf = GridSearchCV(
        estimator=LogisticRegression(penalty='l2'),
        param_grid=parameters,
        cv=sp,
        return_train_score=True)
    # search the best parameters for three different classifiers
    inner_clf_1 = inner_clf

    # select features
    selected_voxels_clf_1 = SelectFpr(f_classif, alpha=0.05).fit(x, y)
    n_features_clf_1 = sum(selected_voxels_clf_1.pvalues_ < 0.05)
    print('You have selected {} features based on Fpr for {}'.format(n_features_clf_1, sub))

    # select the best hyperparameters for clf 1
    inner_clf_1.fit(selected_voxels_clf_1.transform(x), y)
    C_best_clf_1 = inner_clf_1.best_params_['C']
    print('The best C in current cv for clf 1 is', C_best_clf_1)

    # Train the classifier with the best hyperparameter using training and validation set
    classifier_1 = LogisticRegression(penalty='l2', C=C_best_clf_1)
    trained_clf_1 = classifier_1.fit(selected_voxels_clf_1.transform(x), y)

    print('You have trained classifier for {}'.format(sub))

    # save results for each subjects
    results[sub] = {}
    results[sub]['clf_1'] = {}
    results[sub]['clf_1']['C_best'] = C_best_clf_1
    results[sub]['clf_1']['selected_features'] = selected_voxels_clf_1
    results[sub]['clf_1']['n_features'] = n_features_clf_1
    results[sub]['clf_1']['clf_trained'] = trained_clf_1

    # save sub-level results to the disk
    if data_format == 'wholebrain':
        f = open('study_trained_classifier_wholebrain_{}'.format(sub), "wb")
    elif data_format == 'masked':
        f = open('study_trained_classifier_masked_{}'.format(sub), "wb")
    pickle.dump(results[sub], f)
    f.close

# save all subjects data into disk
if data_format == 'wholebrain':
    f = open('study_trained_classifier_wholebrain_all', "wb")
elif data_format == 'masked':
    f = open('study_trained_classifier_masked_all', "wb")
pickle.dump(results, f)
f.close

