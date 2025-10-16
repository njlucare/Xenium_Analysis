#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 16:06:27 2025

@author: nlucarelli
"""
import os
import numpy as np
import tifffile as ti
import pandas as pd
import matplotlib.pyplot as plt
from boundaries_to_mask import BoundaryCSV
from Label_nuc import Nuc
from utils import rotate_contours
from glob import glob
from nuc_feature_contours import Paired_Data
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")

#IU = rot90, flipud
#WashU = 3*rot90

flip = 2
rot = 1
exp_factor = 1.0#0.47

case_directory = '/orange/pinaki.sarder/nlucarelli/Xenium/R01_2/Coords/'
cases = glob(case_directory + '*/')
i=0

reg_pcts = {}

for case in tqdm(cases):
    # case = cases[21]

    case_id = case.split('__')[2].upper()
    case_id = case_id.replace("_", "-")
    
    try:
        numb = int(case_id.split('IU')[-1])
        if numb < 50:
            print(f'Good to go: {numb}')
            flip = 2
            rot = 1
            exp_factor = 1.0#0.47
            if numb==11:
                exp_factor=0.555
            elif numb==5:
                exp_factor=0.555
            elif numb==7:
                exp_factor=0.555
        else:
            print(f'Not gonna work: {numb}')
            continue
    except:
        numb = case_id.split('IU')[-1]
        if ((numb=='F59')|(numb=='K2300080_6PB')):
            print(f'Skipping: {numb}')
            continue
        print(f'Gonna be a big one: {numb}')
        flip = 2
        rot = 1
        exp_factor = 0.555#0.47
        
    
    case_short = case.split('/')[-2]
    
    if os.path.exists(case+'registration.tif'):     
        print(f'Alread registered, skipping: {case_short}')
        continue

    print(f'Working on {case}')

    csv = glob(case+'*nucleus_boundaries.csv.gz')[0]

    dapi_obj = BoundaryCSV(csv,None)
    mask,contours,df = dapi_obj.get_mask()

    # mask=ti.imread(case+'dapi_segmentation.tif')

    ti.imwrite(case+'dapi_segmentation.tif',mask,photometric='minisblack')

    df.to_csv(case+'objects.csv')
    
    
    hes = glob(''.join([case_directory.split('/')[x]+'/' for x in range(len(case_directory.split('/'))-2)])+'*'+case_id+'*')


    he_filename = [x for x in hes if '_hem' in x]
    if len(he_filename)>1:
        he_filename = min(he_filename, key=len)
    else:
        he_filename = he_filename[0]
        
    assert isinstance(he_filename,str)

    # he_filename = glob(case + '*hem.tif')[0]
    he = ti.imread(he_filename)
    

    nuc_obj = Nuc(mask)

    tf_mat,(a,b,c,d),(e,f,g,h) = nuc_obj.get_tf_mat(he,flip=flip,rot=rot,exp_factor=exp_factor)

    full_save = np.vstack((tf_mat,(flip,rot),(exp_factor,0),(a,b),(c,d),(e,f),(g,h)))
    full_save = pd.DataFrame(full_save)
    full_save.to_csv(case + 'tfmat.csv')

    rr = nuc_obj.register(mask, tf_mat,pad_dap=(a,b,c,d),pad_hem=(e,f,g,h),flip=flip,rot=rot,exp_factor=exp_factor)

    ti.imwrite(case+'registration.tif',rr,photometric='minisblack')
    
    he_sum = np.sum(he>0)
    reg_sum = np.sum((he>0)&(rr>0))
    
    reg_pct = reg_sum / he_sum
    
    reg_pcts[case_short] = reg_pct

    # csv2 = glob(case+'*cell_boundaries.csv.gz')[0]
    # cell_obj = BoundaryCSV(csv2, dapi)



    # raise SystemExit
    # wsi = glob(case+'*.svs')[0]
    # full_save = pd.read_csv(case+'tfmat.csv')
    # mask=ti.imread(case+'dapi_segmentation.tif')
    # data_obj = Paired_Data(wsi,csv,np.array(full_save)[:,1:],mask.shape)

    # cyto = ti.imread(case + 'registration.tif')

    # feats = data_obj.get_shape_features()
    # feats2 = data_obj.get_intensity_features()

    # mask2,he2 = data_obj.test_contour(cyto=None)

df = pd.DataFrame(reg_pcts,index=list(range(len(reg_pcts))))
df.to_csv(case_directory+'percentages.csv')