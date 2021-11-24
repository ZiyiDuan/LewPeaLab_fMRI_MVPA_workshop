# -*- coding: utf-8 -*-

"""
@author: Zoe Duan
@file: #clearmen_mask_rawdata.py
@time: 11/5/21 5:07 下午
@desp: mask and concatenate data from different runs. get a final numpy datafile for each sub
    each row represents each TR timepoint, each column represents each voxel
"""



import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
# Import fMRI and general analysis libraries
import nibabel as nib
from nilearn import input_data

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
Load mask
'''
cwd = os.getcwd()

datadir = cwd + '/BIDS/'

mask = nib.load(datadir + 'VVS_mask_bin.nii.gz')
nifti_masker = input_data.NiftiMasker(mask, standardize=True, memory="nilearn_cache")

sub_ids = [27, 34, 44, 77]

for sub_id in sub_ids:
    sub = 'sub-%.3d' % (sub_id)

    img_data_masked = []
    for run_id in range(1, n_runs+1):
        filename = datadir + '{}/func/{}_task-{}_run-{}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub, sub, task, run_id)

        # load the array proxy to save memory and time
        img = nib.load(filename)

        # apply the mask to raw data and get 2D data, each row represents a timepoint, each column represents a voxel
        img_masked = nifti_masker.fit_transform(img)

        # delete the initial 10 timepoints for each run
        img_final = img_masked[10:]

        # stack data in sequence vertically
        if img_data_masked == []:
            img_data_masked = img_final
        else:
            img_data_masked = np.vstack((img_data_masked, img_final))

        del img, img_masked, img_final
    

    print('image data shape for {} is {}'.format(sub, img_data_masked.shape))

    # save data as numpy array
    np.save(datadir + '{}_task-{}_allruns_masked.npy'.format(sub, task), img_data_masked)