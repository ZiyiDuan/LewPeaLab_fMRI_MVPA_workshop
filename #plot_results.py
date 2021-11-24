# -*- coding: utf-8 -*-

"""
@author: Zoe Duan
@file: #plot_results.py
@time: 11/11/21 2:25 下午
@desp: plot results fpr clearmen
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

# Import plotting library
import matplotlib.pyplot as plt
import seaborn as sns


# define some settings
sub_ids = [27, 34, 44, 77]

# data_format = 'wholebrain'


# get current path
cwd = os.getcwd()


'''
plot classification results for category in localizer task
with masked data
'''
# define data format
data_format = 'masked'
# define classify three categories
categories = ['face', 'fruit', 'scene']

# load results
dir = cwd + '/localizer_category_classification/'

f = open(dir + 'classification_results_localizer_masked_all', 'rb')
results_localizer_category = pickle.load(f)
f.close()


# combine all subjects' results and plot the confusion matrix
# get all confusion matrix
clf_ind = 'clf_1'
confusion_matrix_all = {}
confusion_matrix_mean = np.array([])
for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    confusion_matrix_all[sub] = results_localizer_category[sub][clf_ind]['confusion matrix']
    if len(confusion_matrix_mean) == 0:
        confusion_matrix_mean = confusion_matrix_all[sub]
    else:
        confusion_matrix_mean = confusion_matrix_mean + confusion_matrix_all[sub]

confusion_matrix_mean = confusion_matrix_mean/len(sub_ids)


# plot group average results
fig_title = '{} confusion matrix for {} subjects'.format(data_format, len(sub_ids))
fig_name = dir + '{}_confusion_matrix_{}_{}_subjects.png'.format(data_format, clf_ind, len(sub_ids))


f, ax = plt.subplots(1, 1, figsize=(15, 12))
ax.set_title(fig_title, fontsize=30)
sns.heatmap(confusion_matrix_mean, vmin=0, vmax=1, cmap='YlGnBu', linewidths=0.2,
            annot=True, annot_kws={"fontsize":30}, cbar=True)
ax.set_xticklabels(labels=categories, fontsize=30)
ax.set_yticklabels(labels=categories, fontsize=30)
ax.set_xlabel('Predicted category', fontsize=30)
ax.set_ylabel('Actual category', fontsize=30)
plt.savefig(fig_name, bbox_inches='tight')
plt.show()


# plot subject-level results
for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    confusion_matrix_i = confusion_matrix_all[sub]

    # plot the confusion matrix
    if data_format == 'wholebrain':
        fig_title = 'wholebrain confusion matrix for {}'.format(sub)
        fig_name = dir + 'wholebrain_confusion_matrix_{}_{}.png'.format('clf_1', sub)
    elif data_format == 'masked':
        fig_title = 'masked confusion matrix for {}'.format(sub)
        fig_name = dir + 'maked_confusion_matrix_{}_{}.png'.format('clf_1', sub)

    f, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.set_title(fig_title, fontsize=30)
    sns.heatmap(confusion_matrix_i, vmin=0, vmax=1, cmap='YlGnBu', linewidths=0.2,
                annot=True, annot_kws={"fontsize":30}, cbar=True)
    ax.set_xticklabels(labels=categories, fontsize=30)
    ax.set_yticklabels(labels=categories, fontsize=30)
    ax.set_xlabel('Predicted category', fontsize=30)
    ax.set_ylabel('Actual category', fontsize=30)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()









'''
plot classification results for subcategory in localizer task
with masked data
'''
# define data format
data_format = 'masked'
# classify nine sub-categories
categories = ['actor', 'musician', 'politician', 'apple', 'grape', 'pear', 'beach', 'bridge', 'mountain']

# load results
dir = cwd + '/localizer_subcategory_classification/'

f = open(dir + 'classification_results_localizer_masked_all', 'rb')
results_localizer_category = pickle.load(f)
f.close()


# combine all subjects' results and plot the confusion matrix
# get all confusion matrix
clf_ind = 'clf_1'
confusion_matrix_all = {}
confusion_matrix_mean = np.array([])
for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    confusion_matrix_all[sub] = results_localizer_category[sub][clf_ind]['confusion matrix']
    if len(confusion_matrix_mean) == 0:
        confusion_matrix_mean = confusion_matrix_all[sub]
    else:
        confusion_matrix_mean = confusion_matrix_mean + confusion_matrix_all[sub]

confusion_matrix_mean = confusion_matrix_mean/len(sub_ids)


# plot group average results
if data_format == 'wholebrain':
    fig_title = 'wholebrain confusion matrix for {} subjects'.format(len(sub_ids))
    fig_name = dir + 'wholebrain_confusion_matrix_{}_{}_subjects.png'.format(clf_ind, len(sub_ids))
elif data_format == 'masked':
    fig_title = 'masked confusion matrix for {} subjects'.format(len(sub_ids))
    fig_name = dir + 'masked_confusion_matrix_{}_{}_subjects.png'.format(clf_ind, len(sub_ids))

f, ax = plt.subplots(1, 1, figsize=(15, 12))
ax.set_title(fig_title, fontsize=30)
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


# plot subject-level results
for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    confusion_matrix_i = confusion_matrix_all[sub]

    # plot the confusion matrix
    fig_title = '{} confusion matrix for {} subjects'.format(data_format, len(sub_ids))
    fig_name = '{}_confusion_matrix_{}_{}_subjects.png'.format(data_format, clf_ind, len(sub_ids))

    f, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.set_title(fig_title, fontsize=30)
    sns.heatmap(confusion_matrix_i, vmin=0, vmax=0.4, cmap='YlGnBu', linewidths=0.2,
                annot=True, annot_kws={"fontsize": 15}, cbar=True)
    ax.set_xticklabels(labels=categories, fontsize=20, rotation=90)
    ax.set_yticklabels(labels=categories, fontsize=20, rotation=0)
    ax.set_xlabel('Predicted category', fontsize=30)
    ax.set_ylabel('Actual category', fontsize=30)
    # ax.xaxis.set_label_position('top')
    # ax.xaxis.tick_top()
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()





'''
plot timecourses classification results for category in study task
with masked data
'''

