#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:07:06 2025

@author: nlucarelli
"""
import os
import numpy as np
import tifffile as ti
import tiffslide as openslide
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from nuc_feature_contours import Paired_Data
import warnings

warnings.filterwarnings('ignore')

case_directory = '/orange/pinaki.sarder/nlucarelli/Xenium/R01_2/Coords/'
cases = glob(case_directory + '*/')
i=0

for case in cases:

    if not os.path.exists(case+'registration.tif'):
        print(f'Registration not done, skipping {case}')
        continue
    
    if os.path.exists(case+'nuc_features.csv'):
        print(f'Feature extraction done, skipping {case}')
        continue

    csv = glob(case+'*nucleus_boundaries.csv*')[0]
    
    # mask = ti.imread(case+'registration.tif')
    mask = ti.imread(case+'dapi_segmentation.tif')
    
    # print(mask.shape)
    # print(mask2.shape)
    
    case+'tfmat.csv'

    full_save = pd.read_csv(case+'tfmat.csv')
    
    case_id = case.split('__')[2].upper()
    case_id = case_id.replace("_", "-")
    
    hes = glob(''.join([case_directory.split('/')[x]+'/' for x in range(len(case_directory.split('/'))-2)])+'*'+case_id+'*')
    
    he_filename = [x for x in hes if '_converted' in x]
    if len(he_filename)>1:
        he_filename = min(he_filename, key=len)
    else:
        he_filename = he_filename[0]

    wsi = he_filename#glob(case+'*.ndpi')[0]
    
   
    
    data_obj = Paired_Data(wsi,csv,np.array(full_save)[:,1:],mask.shape,filter_csv=None)
    
    # nu,h = data_obj.test_contour('ifioaeae-1')
                
    features = data_obj.get_all_features()#types=cell_types)
    features.to_csv(case+'nuc_features.csv')


    csv2 = glob(case+'*cell_boundaries.csv*')[0]

    cell_obj = Paired_Data(wsi,csv2,np.array(full_save)[:,1:],mask.shape,filter_csv=None)

    
    features2 = cell_obj.get_all_features()#types=cell_types)
    features2.to_csv(case+'cell_features.csv')
