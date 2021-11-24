# -*- coding: utf-8 -*-

"""
@author: Zoe Duan
@file: #clearmen_RSA.py
@time: 11/14/21 3:32 PM
@desp: decode the neural representation of individual stimuli in study task
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
from nilearn.plotting import plot_design_matrix
# %matplotlib notebook

# Import machine learning libraries
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest, SelectFpr
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import sem
from copy import deepcopy
from sklearn.metrics import confusion_matrix

from nilearn.glm.first_level import FirstLevelModel
from nilearn.signal import clean
from nilearn.image import mean_img
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_contrast_matrix
from nilearn.plotting import plot_stat_map

# We still have to import the functions of interest
from utils import shift_timing, normalize
from config import pad_contrast



'''
Set experimental parameters
'''
# define which task the data comes from
task = 'localizer'
# task = 'study'

TR = 0.46
shift_size = 10

if task == 'localizer':
    n_runs = 5
elif task == 'study':
    n_runs = 6

# classify three categories
# categories_index = [0, 1, 2, 3]
categories = ['rest', 'face', 'fruit', 'scene']

cwd = os.getcwd()
datadir = cwd + '/BIDS/'

# define output folder, if not exist, create one
outdir = cwd + '/localizer_RSA/'
if not os.path.exists(outdir):os.makedirs(outdir, exist_ok=True)




'''
load localizer design matrix
'''
design_matrix = pd.read_csv(datadir + 'localizer_design_matrix_new.csv')
# The new design matrix changed the category and subcategory label into 0 when its rest states

# get labels
run_labels = np.array(design_matrix['run']).reshape(len(design_matrix['run']), 1)
category_label_raw = np.array(design_matrix['category']).reshape(len(design_matrix['category']), 1)
subcategory_label_raw = np.array(design_matrix['subcategory']).reshape(len(design_matrix['subcategory']), 1)
image_id_raw = np.array(design_matrix['image_id']).reshape(len(design_matrix['image_id']), 1)
stimulus_raw = np.array(design_matrix['stimulus']).reshape(len(design_matrix['stimulus']), 1)
# 0=rest, 1=stimulus presented


'''
load localizer trial design index
'''
trial_design = pd.read_csv(datadir + 'localizer_trial_design_index.csv')
run_labels_trial = trial_design['run']

events_category = []
for run in np.unique(run_labels_trial):
    curr_events = {}
    curr_events['onset'] = trial_design.loc[trial_design['run']==run, 'onset_sec']
    curr_events['duration'] = trial_design.loc[trial_design['run']==run, 'duration']
    curr_events['trial_type'] = trial_design.loc[trial_design['run']==run, 'category']
    df = pd.DataFrame(curr_events)
    events_category.append(df)



'''
load the data
'''
# load mask
mask = nib.load(datadir + 'VVS_mask_bin.nii.gz')

# load raw img data
sub_ids = [27]
# sub_ids = [27, 34, 44, 77]

# pre-allocate space to store image data
run_imgs_all = {}
# pre-allocate space to store confounds
confounds_all = {}

for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)

    run_imgs_id = []
    confounds_id = []
    for run_id in range(1, n_runs+1):
        # load image data
        img_filename = datadir + '{}/func/{}_task-{}_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, sub, task, run_id)
        # load the array proxy to save memory and time
        img_ = nib.load(img_filename)
        run_imgs_id.append(img_)

        # load confounds file
        confounds_filename = datadir + '{}/func/{}_task-{}_run-{}_desc-confounds_timeseries.tsv'.format(sub, sub, task, run_id)
        confounds_ = pd.read_csv(confounds_filename, sep='\t')
        # find only motion confounds
        motion_cols = [col for col in confounds_.columns if 'motion' in col]
        print('All columns names: ', list(confounds_.columns))
        print('Only motion columns names: ', motion_cols)

        # save motion confounds
        confounds_motion = confounds_[motion_cols]
        confounds_id.append(confounds_motion)


    run_imgs_all[sub] = run_imgs_id
    confounds_all[sub] = confounds_id

    print('You have loaded {} data and confounds from {}'.format(task, sub))




'''
category-level feature selection by using GLM
'''

# define linear model
model = FirstLevelModel(t_r=TR, slice_time_ref=.5, hrf_model='spm',
                        drift_model=None, high_pass=None, mask_img=mask, signal_scaling=False,
                        smoothing_fwhm=6, noise_model='ar1', n_jobs=-1, verbose=2)


for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)
    run_imgs = run_imgs_all[sub]
    confounds = confounds_all[sub]

    # build up category-level feature selection mask
    # fit category-level GLM
    model.fit(run_imgs=run_imgs, events=events_category, confounds=confounds)

    print('You have fit the GLM for {}'.format(sub))

    # get the design matrix for all runs
    desing_matrix_category = model.design_matrices_

    # plot and save the design matrix for each run
    for run_id in range(n_runs):
        curr_design_matrix = desing_matrix_category[run_id]

        # plot current design matrix
        f, ax = plt.subplots(1, 1)
        ax.set_title('Design matrix for run {}'.format(run_id + 1), fontsize=30)
        plot_design_matrix(curr_design_matrix, ax=ax)
        plt.show()

        # save design matrix
        plot_design_matrix(curr_design_matrix, output_file=os.path.join(outdir, '{}_run_{}_design_matrix.png'.format(sub, run_id+1)))


    # plot the expected response profile of regions sensitive to face, fruit, scene
    # use run 1 as an example
    f, ax = plt.subplots(3, 1, figsize=(20, 18))

    ax[0].plot(desing_matrix_category[0][1])
    ax[0].set_xlabel('scan')
    ax[0].set_title('Expected Face Response')

    ax[1].plot(desing_matrix_category[0][2])
    ax[1].set_xlabel('scan')
    ax[1].set_title('Expected Fruit Response')

    ax[2].plot(desing_matrix_category[0][3])
    ax[2].set_xlabel('scan')
    ax[2].set_title('Expected Scene Response')

    plt.savefig(outdir+'{}_run_1_Expected_response.png'.format(sub), bbox_inches='tight')
    plt.show()


    # grab the number of regressors in the model for each run
    n_columns = model.design_matrices_[0].shape[-1]

    # define the contrasts
    '''the order of trial types is stored in model.design_matrices_[0].columns
        pad_contrast() adds 0s to the end of a vector in the case that other regressors are modeled, 
        but not included in the primary contrasts
    '''
    contrasts = {
        'face_VS_fruit_&_scene': pad_contrast([0, 2, -1, -1], n_columns),
        'fruit_VS_face_&_scene': pad_contrast([0, -1, 2, -1], n_columns),
        'scene_VS_face_&_fruit': pad_contrast([0, -1, -1, 2], n_columns)
    }

    # plot the contrast
    # use run 1 as an example
    for contrast in contrasts:
        plot_contrast_matrix(contrasts[contrast], design_matrix=desing_matrix_category[0])
        plt.show()

    # compute and save contrasts
    z_map_all_contrasts = {}
    for contrast in contrasts:
        z_map = model.compute_contrast(contrasts[contrast], output_type='z_score')
        z_map_all_contrasts[contrast] = z_map
        nib.save(z_map, os.path.join(outdir, sub + '_' + contrast + '_zmap.nii.gz'))

    # select mask according to statistical significance testing
    for contrast in contrasts:
        z_map = z_map_all_contrasts[contrast]
        category_map, threshold = threshold_stats_img(
            z_map, alpha=.05, height_control=None, cluster_threshold=10)
        plot_stat_map(category_map, bg_img=None, threshold=threshold,
                      display_mode='z', cut_coords=3, black_bg=True,
                      title=contrast+'\nUncorrected P <.05, clusters > 10 voxels')
        plt.savefig(outdir + '{}_{}_category_contrast_map.png'.format(sub, contrast), bbox_inches='tight')
        plt.show()

