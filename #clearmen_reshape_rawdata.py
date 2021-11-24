# -*- coding: utf-8 -*-

"""
@author: Zoe Duan
@file: #clearmen_reshape_rawdata.py
@time: 10/29/21 11:36 上午
@desp: reshape and concatenate data from different runs. get a final numpy datafile for each sub
    each row represents each TR timepoint, each column represents each voxel
@data:
    preprocessed four subjects' data, organized in BIDS
    TR = 460ms
Functional localizer task:
    5 runs, 3 categories with 3 subcategories for each, 90 trials each category, 270 trials in total
    Each run: 54 trials, 9 trials grouped into a subcategory-specific triplets, 6 triplets for each category
            13 TRs after each triplet, 13 TRs before each run
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


'''
Set experimental parameters
'''
# define which task the data comes from
# task = 'localizer'
task = 'study'


TR = 0.46

if task == 'localizer':
    n_runs = 5
elif task == 'study':
    n_runs = 6


'''
Load the metadata
'''
cwd = os.getcwd()

datadir = cwd + '/BIDS/'

# sub_ids = [44]
sub_ids = [27, 34, 44, 77]

for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)

    img_data_raw = []
    for run_id in range(1, n_runs+1):
        filename = datadir + '{}/func/{}_task-{}_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, sub, task, run_id)

        # load the array proxy to save memory and time
        img = nib.load(filename)
        # check the state of the cache
        print('Whether the img is in memory now?', img.in_memory)

        # load the data into memory
        img_data = img.get_fdata()
        print('Whether the img is in memory now?', img.in_memory)


        # reshape the data into 2-D, each row represents a voxel, each column represents a timepoint
        img_data_reshaped = np.reshape(img_data, (img.shape[0]*img.shape[1]*img.shape[2], img.shape[3]))
        img_data_reshaped = img_data_reshaped.T

        # delete the initial 10 timepoints for each run
        img_data_final = img_data_reshaped[10:]

        # stack data in sequence vertically
        if img_data_raw == []:
            img_data_raw = img_data_final
        else:
            img_data_raw = np.vstack((img_data_raw, img_data_final))

        # delete data to save memory
        img.uncache()
        del img, img_data, img_data_reshaped, img_data_final

    print('image data shape for {} is {}'.format(sub, img_data_raw.shape))

    # save data as numpy array
    np.save(datadir + '{}_task-{}_allruns_raw.npy'.format(sub, task), img_data_raw)





