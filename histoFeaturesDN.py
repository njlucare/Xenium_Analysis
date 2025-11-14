#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:52:37 2025

@author: nlucarelli
"""
import os
import numpy as np
import tifffile as ti
import pandas as pd
import tiffslide as openslide
import matplotlib.pyplot as plt
import seaborn as sns
from PAS_deconvolution2 import deconvolution_WSI
from skimage.filters import threshold_otsu
from glob import glob
import warnings


# sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

def save_im(im,fn):
    from PIL import Image
    
    im = Image.fromarray(im)
    im.save(fn)

class Mat:
    def __init__(self,mat):
        self.mat = mat
    
    def plot_features(self):
        colors = np.vectorize(self.__color_mapping)(self.mat)
        plt.figure(figsize=(16,5))#3.75x2.75 for 3 rows, 3.75x2.5 for 2 rows
        sns.set(font_scale=1.3)
        sns.heatmap(colors,cmap="coolwarm",yticklabels=cell_types,xticklabels=feature_names,linewidth=0.5,linecolor='black',vmin=-1,vmax=1)#,annot=self.mat,fmt='.2f')
        plt.ylabel("cell type",fontsize=12)
        plt.xlabel("",fontsize=12)
        plt.xticks(fontsize=12,rotation=45,ha='right',rotation_mode='anchor')
        plt.yticks(fontsize=12,rotation=45,ha='right',rotation_mode='anchor')
        plt.tight_layout()
    
    
    def __color_mapping(self,value):
        # row,col = np.where(self.mat==value)
        # true_label = col[0]#row[0]
        # class_sizes = np.sum(self.mat,axis=0)#axis=1
        # color_intensity = value / class_sizes[true_label]
        
        return value#color_intensity
        
        # try:
        #     row,col = np.where(self.mat==value)
        #     true_label = row[0]#col[0]
        #     # class_mean = np.mean(self.mat,axis=1)[true_label]
        #     # class_std = np.std(self.mat,axis=1)[true_label]
        #     color_intensity = (value - mns[true_label]) / stds[true_label]   
            
        #     color_intensity = (value-mmin[true_label])/(mmax[true_label]-mmin[true_label])
        

        #     return color_intensity#value
        
        # except:
        #     return 0#value
    
    

MODx=np.zeros((3,))
MODy=np.zeros((3,))
MODz=np.zeros((3,))
MODx[0]= 0.634
MODy[0]= 0.669
MODz[0]= 0.389

MODx[1]= 0.507
MODy[1]= 0.725
MODz[1]= 0.465

MODx[2]= 0.182
MODy[2]= -0.168
MODz[2]= 0.765

MOD=[MODx,MODy,MODz]

warnings.filterwarnings('ignore')

case_directory = '/orange/pinaki.sarder/nlucarelli/Xenium/R01_2/Coords/'
cases = glob(case_directory + '*/')
i=0

annotations = []

for case in cases:

    if not os.path.exists(case+'registration.tif'):
        print(f'Registration not done, skipping {case}')
        continue

    case_id = case.split('__')[2].upper()
    case_id = case_id.replace("_", "-")
    
    annot_filename = glob(''.join([case.split('/')[x]+'/' for x in range(len(case_directory.split('/')))])+'*annot*') 
    
    if len(annot_filename)>1:
        annot_filename = min(annot_filename, key=len)
    else:
        annot_filename = annot_filename[0]
    
    annotation = pd.read_csv(annot_filename)
    labs = [case_id for x in range(len(annotation))]
    
    annotation['case_id'] = labs
    annotations.append(annotation)

annotations = pd.concat(annotations,axis=0)

features_raw = pd.read_csv('/orange/pinaki.sarder/nlucarelli/Xenium/R01_2/features/features_full.csv')
features_raw.loc[features_raw['case_name']=='IU20','cohort'] = 'ref'
feature_names = list(features_raw.columns[5:34])

cell_types = sorted(list(set(list(annotations['group']))))

cell_types_dict = {
    'EC': ['EC-GC','EC-GC (DKD)'],
    'TAL': ['TAL','altTAL'],
    'IMM': ['B','DC','Lymphoid','MAST','MON','Myeloid','N','NEU','NK','PL','T','moMAC','moMAC-INF','resMAC','FIB','aFIB','infFIB','pvFIB','MYOF'],
    'FIB': ['FIB','aFIB','infFIB','pvFIB','MYOF'],
    'MYOF': ['MYOF'],
    'GLOM': ['EC-GC','EC-GC (DKD)','MC','POD','PEC'],
                   }

key_oi = 'EC'
obj_oi = 'nuc'

cell_types = cell_types_dict[key_oi]

features_raw = features_raw[features_raw['type']==obj_oi]
features = pd.merge(annotations,features_raw,on='cell_id',how='inner')

del annotations,features_raw

feature_means = features[features['group'].isin(cell_types)]

feature_means = feature_means.loc[:,feature_names]

feature_means = np.float32(feature_means)

mmax = np.max(feature_means,axis=0)
mmin = np.min(feature_means,axis=0)

mns = np.mean(feature_means,axis=0)
stds = np.std(feature_means,axis=0)

feature_values = np.zeros((len(cell_types),len(feature_names)))
feature_stds = np.zeros((len(cell_types),len(feature_names)))

for j in range(len(cell_types)):
    
    for k in range(len(feature_names)):
        
        cell_subset = features[features['group']==cell_types[j]]
        cell_subset = cell_subset.loc[:,feature_names]
        cell_subset = np.float32(cell_subset)
        
        feature_values[j,k] = (np.mean(cell_subset[:,k])-mns[k])/stds[k]
        feature_stds[j,k] = np.std(cell_subset[:,k])/stds[k]


obj = Mat(feature_values)
obj.plot_features()
plt.savefig(f'/orange/pinaki.sarder/nlucarelli/Xenium/R01_2/plots/heatmaps/features_{key_oi}_{obj_oi}.png',dpi=300)
plt.close()


     
