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

def get_tn(img):
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.color import rgb2hsv
    from skimage.filters import gaussian,threshold_otsu
    from skimage.morphology import remove_small_objects
    from PIL import Image
    

    thumbIm = np.array(img)
    x,y,_ = thumbIm.shape
    thumbIm = thumbIm[::16,::16,0:3]
    hsv=rgb2hsv(thumbIm)
    g=gaussian(hsv[:,:,1],1)
    binary = g>threshold_otsu(g)
    # binary=(g>0.1).astype('bool')
    # binary=binary_fill_holes(binary)
    # binary_save = 255*np.uint8(binary)
    
    # binary_save = Image.fromarray(binary_save)
    
    binary = remove_small_objects(binary,10000)
    binary = Image.fromarray(binary)
    
    binary = binary.resize((y, x), Image.Resampling.NEAREST)
    

    return np.array(binary)


warnings.filterwarnings("ignore")

#IU = rot90, flipud
#WashU = 3*rot90

flip = 0
rot = 3
exp_factor = 1.0#0.47

case_directory = '/orange/pinaki.sarder/Davy_Jones_Locker/SWAT/'
cases = glob(case_directory + '*/*/')
i=0

cases = [case for case in cases if 'json' not in case]

reg_pcts = {}

skip_idxs = [
             0,1,2,3,4,5,6,#kidney
             7,8,9,10,11,#lung
             12,13,14,15,16,17,18,19,20,#intestine
             # 21,22,23,24,25,26,27,28,29,30,31,32,33,#skin
             ]

small_sizes = [True,True,True,True,True,True,True,
               True,True,True,True,True,
               True,True,True,True,True,True,True,True,True,
               False,False,False,False,False,False,False,False,False,False,False,False,False,
               ]

cases = [cases[25]]
i=25

for case in tqdm(cases):
    
    if i in skip_idxs:
        print(f'Dont like this section, skipping {case}')
        i+=1
        continue

    case_id = case.split('__')[2].upper()
    case_id = case_id.replace("_", "-")
    
    small_size = small_sizes[i]
    
    # try:
    #     numb = int(case_id.split('IU')[-1])
    #     if numb < 50:
    #         print(f'Good to go: {numb}')
    #         flip = 2
    #         rot = 1
    #         exp_factor = 1.0#0.47
    #         if numb==11:
    #             exp_factor=0.555
    #         elif numb==5:
    #             exp_factor=0.555
    #         elif numb==7:
    #             exp_factor=0.555
    #     else:
    #         print(f'Not gonna work: {numb}')
    #         continue
    # except:
    #     numb = case_id.split('IU')[-1]
    #     if ((numb=='F59')|(numb=='K2300080_6PB')):
    #         print(f'Skipping: {numb}')
    #         continue
    #     print(f'Gonna be a big one: {numb}')
    #     flip = 2
    #     rot = 1
    #     exp_factor = 0.555#0.47
        
    
    case_short = case.split('/')[-2]
    
    # if os.path.exists(case+'registration.tif'):     
    #     print(f'Alread registered, skipping: {case_short}')
    #     i+=1
    #     continue

    print(f'Working on {case}')
    print(f'Index = {i}')

    csv = glob(case+'*nucleus_boundaries.csv.gz')[0]

    dapi_obj = BoundaryCSV(csv,None)#,mpp=1.0)
    mask,contours,df = dapi_obj.get_mask()

    # mask=ti.imread(case+'dapi_segmentation.tif')

    # ti.imwrite(case+'dapi_segmentation.tif',mask,photometric='minisblack')

    # df.to_csv(case+'objects.csv')
    
    
    # hes = glob(''.join([case_directory.split('/')[x]+'/' for x in range(len(case_directory.split('/'))-2)])+'*'+case_id+'*')
    hes = glob(case+'*_mask.tif')#_hem.tif

    he_filename = [x for x in hes if '_mask' in x]
    if len(he_filename)>1:
        he_filename = min(he_filename, key=len)
    elif len(he_filename)==0:
        print(f'Cant find hematoxylin file for {case_id}, skipping...')
        i+=1
        continue
    else:
        he_filename = he_filename[0]
        
    assert isinstance(he_filename,str)

    # he_filename = glob(case + '*hem.tif')[0]
    he = ti.imread(he_filename)
    
    # import tiffslide as openslide
    # s = openslide.OpenSlide(he_filename.split('_hem')[0]+'.svs')
    # xz,yz = s.dimensions
    # he = s.read_region((0,0),0,(xz,yz))
    # he = get_tn(he)
    # he = 255*he.astype(np.uint8)
    # ti.imwrite('/orange/pinaki.sarder/Davy_Jones_Locker/SWAT/Skin/output-XETG00102__0060796__4B__20250502__183311/tn.tif',he,photometric='minisblack')
    # raise SystemExit#
    
    
    if small_size:
        mask = mask[::2,::2]
        he = he[::2,::2]
    

    nuc_obj = Nuc(mask)

    tf_mat,(a,b,c,d),(e,f,g,h) = nuc_obj.get_tf_mat(he,flip=flip,rot=rot,exp_factor=exp_factor)

    full_save = np.vstack((tf_mat,(flip,rot),(exp_factor,0),(a,b),(c,d),(e,f),(g,h),(small_size,0)))
    full_save = pd.DataFrame(full_save)
    full_save.to_csv(case + 'tfmat.csv')

    rr = nuc_obj.register(mask, tf_mat,pad_dap=(a,b,c,d),pad_hem=(e,f,g,h),flip=flip,rot=rot,exp_factor=exp_factor,small_size=small_size)

    ti.imwrite(case+'registration.tif',rr,photometric='minisblack')
    
    he_sum = np.sum(he>0)
    # reg_sum = np.sum((he>0)&(rr>0))
    
    # reg_pct = reg_sum / he_sum
    
    # reg_pcts[case_short] = reg_pct

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
    i+=1

# df = pd.DataFrame(reg_pcts,index=list(range(len(reg_pcts))))
# df.to_csv(case_directory+'percentages.csv')