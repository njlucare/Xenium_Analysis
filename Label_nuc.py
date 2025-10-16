#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tifffile as ti
import multiprocessing
from joblib import Parallel, delayed, Memory
from pystackreg import StackReg
from glob import glob
from tqdm import tqdm
from PAS_deconvolution2 import deconvolution_WSI
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.filters import threshold_local, threshold_otsu
from skimage.transform import resize as rsz
from scipy.ndimage import zoom
from scipy.ndimage import affine_transform

class Nuc:
    def __init__(self,nuc):

        if isinstance(nuc, str):
            self.nuc = ti.imread(nuc)
        else:
            self.nuc = nuc
        self.num_processes=multiprocessing.cpu_count()

    def register(self,mat,tf_mat,pad_dap=None,pad_hem=None,flip=None,rot=0,exp_factor=1):

        #.astype(np.uint16)
        
        mat = np.rot90(mat,rot)

        if flip==1:
            mat = np.flipud(mat)
        elif flip==2:
            mat = np.fliplr(mat)
            
        mat = zoom(mat,exp_factor,order=0)

        if pad_dap is not None:
            a,b,c,d=pad_dap
            # pad_dap_h,pad_dap_w = pad_dap
            mat = self.nuc_pad_reg(mat, a,b,c,d)

        if pad_hem is not None:
            # pad_hem_h,pad_hem_w = pad_hem
            e,f,g,h=pad_hem

        tf_mat_normalized = tf_mat[0:2,:]
        offset = np.squeeze(tf_mat[2:,:])

        # print(mat.shape)
        # raise SystemExit

        mask = affine_transform(
            mat,
            tf_mat_normalized,
            offset = offset,
            order=0,
            mode='constant',
            cval=0
            )

        return self.hem_unpad(mask,e,f,g,h)

    def map_cells(self,clusters,centroids):

        assert centroids.shape[0] == clusters.shape[0]
        assert centroids.shape[1] == 2, 'Must have two centroid columns'

        n_nuc = int(np.max(self.nuc))

        assert n_nuc == clusters.shape[0], 'Difference between image and matrix sizes'

        new_nucs = np.zeros_like(self.nuc)

        new_nucs = Parallel(n_jobs=self.num_processes,prefer="threads",require="sharedmem")(delayed(self.__map_slide)(i,centroids,clusters,self.nuc,new_nucs) for i in tqdm(range(n_nuc),desc='Nucleus #'))
        new_nucs = self.__map_cleanup(new_nucs)

        return new_nucs

    def mask_to_label(self,mask,ids,clusters):


        mask = ti.imread(mask)
        mask_fill = np.zeros_like(mask)


        n_cells = np.max(mask)

        df_ids = pd.read_csv(ids)
        df_clusters = pd.read_csv(clusters)

        cluster_idx = df_clusters.index

        for i in tqdm(range(1,n_cells+1)):
            id_temp = df_ids.loc[df_ids["cell_no"]==i,"cell_id"][i-1]
            try:
                cluster_values = df_clusters.loc[df_clusters["Barcode"]==id_temp,"Cluster"]
                cluster_temp = cluster_values[int(cluster_values.index[0])]
            except:
                print(f'Skipping: {i}')
                cluster_temp=0

            cluster_temp = int(cluster_temp)

            if cluster_temp!=0:
                mask_fill[mask==i] = cluster_temp

        return mask_fill

    def get_tf_mat(self,hem_bin,flip=None,rot=0,exp_factor=1):

        transformations = {
            'AFFINE': StackReg.AFFINE
        }

        nuc_bin = (self.nuc > 0).astype('bool')
        hem_bin = (hem_bin > 0).astype('bool')

        nuc_bin = np.rot90(nuc_bin,rot)

        if flip==1:
            nuc_bin = np.flipud(nuc_bin)
        elif flip==2:
            nuc_bin = np.fliplr(nuc_bin)


        nuc_bin = zoom(nuc_bin,exp_factor,order=0)

        heightHisto, widthHisto = hem_bin.shape
        heightIF, widthIF = nuc_bin.shape

        pad_hem_h = heightIF-heightHisto if heightIF > heightHisto else 0
        pad_hem_w = widthIF-widthHisto if widthIF > widthHisto else 0

        pad_dap_h = heightHisto-heightIF if heightHisto > heightIF else 0
        pad_dap_w = widthHisto-widthIF if widthHisto > widthIF else 0

        #PAD THE IMAGES IF NEEDED

        nuc_bin,(a,b,c,d)=self.nuc_pad(nuc_bin,pad_dap_h,pad_dap_w)
        hem_bin,(e,f,g,h)=self.hem_pad(hem_bin,pad_hem_h,pad_hem_w)
        
        nuc_bin[:,5000:] = 0
        hem_bin[:,5000:] = 0

        for i, (name, tf) in enumerate(transformations.items()):
            sr = StackReg(tf)
            #REGISTER!
            transformation_matrix = sr.register(hem_bin,nuc_bin)
            tf_mat_normalized = transformation_matrix[:2,:2]
            tf_mat_normalized = np.rot90(tf_mat_normalized,k=2)

            offset = transformation_matrix[:2,2]
            offset = np.flip(offset)
            offset = np.expand_dims(offset,0)

        tf_mat = np.concatenate((tf_mat_normalized,offset),axis=0)

        return tf_mat,(a,b,c,d),(e,f,g,h)


    def __map_slide(self,i,cluster_matrix,base_cluster,nucs,new_nucs):
        x_cent = int(cluster_matrix[i,1])
        y_cent = int(cluster_matrix[i,0])

        clus_0 = base_cluster[i]+1

        l = nucs[y_cent,x_cent]

        if l==0:
            return new_nucs
        else:
            new_nucs[nucs==l] = clus_0

            return new_nucs

    def __map_cleanup(self,new_nucs):
        if len(new_nucs)>0:
            new_nucs=new_nucs[0]
        else:
            new_nucs = []
        return new_nucs


    def nuc_pad(self,nuc_bin,pad_dap_h,pad_dap_w):#CHANGE THIS SO EACH IS A VARIABLE, SO I CAN SAVE IT IN TF ARRAY AS WELL
        if (pad_dap_h % 2 != 0) & (pad_dap_w % 2 != 0):
            a,b,c,d = pad_dap_h//2+1,pad_dap_h//2,pad_dap_w//2+1,pad_dap_w//2
            return np.pad(nuc_bin,((pad_dap_h//2+1,pad_dap_h//2),(pad_dap_w//2+1,pad_dap_w//2)),mode='constant'),(a,b,c,d)
            # return np.pad(nuc_bin,((pad_dap_h,0),(pad_dap_w//2+1,pad_dap_w//2)),mode='constant')
        elif (pad_dap_h % 2 != 0) & (pad_dap_w % 2 == 0):
            a,b,c,d = pad_dap_h//2+1,pad_dap_h//2,pad_dap_w//2,pad_dap_w//2
            return np.pad(nuc_bin,((pad_dap_h//2+1,pad_dap_h//2),(pad_dap_w//2,pad_dap_w//2)),mode='constant'),(a,b,c,d)
            # return np.pad(nuc_bin,((pad_dap_h,0),(pad_dap_w//2,pad_dap_w//2)),mode='constant')
        elif (pad_dap_h % 2 == 0) & (pad_dap_w % 2 != 0):
            a,b,c,d = pad_dap_h//2,pad_dap_h//2,pad_dap_w//2+1,pad_dap_w//2
            return np.pad(nuc_bin,((pad_dap_h//2,pad_dap_h//2),(pad_dap_w//2+1,pad_dap_w//2)),mode='constant'),(a,b,c,d)
            # return np.pad(nuc_bin,((pad_dap_h,0),(pad_dap_w//2+1,pad_dap_w//2)),mode='constant')
        else:
            a,b,c,d = pad_dap_h//2,pad_dap_h//2,pad_dap_w//2,pad_dap_w//2
            return np.pad(nuc_bin,((pad_dap_h//2,pad_dap_h//2),(pad_dap_w//2,pad_dap_w//2)),mode='constant'),(a,b,c,d)
            # return np.pad(nuc_bin,((pad_dap_h,0),(pad_dap_w//2,pad_dap_w//2)),mode='constant')
    def nuc_pad_reg(self,nuc_bin,a,b,c,d):
        
        return np.pad(nuc_bin,((a,b),(c,d)),mode='constant')

    def hem_pad(self,hem_bin,pad_hem_h,pad_hem_w):
        # return np.pad(hem_bin,((pad_hem_h//2,pad_hem_h//2),(pad_hem_w//2,pad_hem_w//2)),mode='constant')#93
        # # return np.pad(hem_bin,((pad_hem_h//3+1,2*pad_hem_h//3),(pad_hem_w//2,pad_hem_w//2)),mode='constant')#94
        if (pad_hem_h % 2 != 0) & (pad_hem_w % 2 != 0):
            e,f,g,h = pad_hem_h//2+1,pad_hem_h//2,pad_hem_w//2+1,pad_hem_w//2
            return np.pad(hem_bin,((pad_hem_h//2+1,pad_hem_h//2),(pad_hem_w//2+1,pad_hem_w//2)),mode='constant'),(e,f,g,h)
            # return np.pad(hem_bin,((pad_hem_h,0),(pad_hem_w//2+1,pad_hem_w//2)),mode='constant')
        elif (pad_hem_h % 2 != 0) & (pad_hem_w % 2 == 0):
            e,f,g,h = pad_hem_h//2+1,pad_hem_h//2,pad_hem_w//2,pad_hem_w//2
            return np.pad(hem_bin,((pad_hem_h//2+1,pad_hem_h//2),(pad_hem_w//2,pad_hem_w//2)),mode='constant'),(e,f,g,h)
            # return np.pad(hem_bin,((pad_hem_h,0),(pad_hem_w//2,pad_hem_w//2)),mode='constant')
        elif (pad_hem_h % 2 == 0) & (pad_hem_w % 2 != 0):
            e,f,g,h = pad_hem_h//2,pad_hem_h//2,pad_hem_w//2+1,pad_hem_w//2
            return np.pad(hem_bin,((pad_hem_h//2,pad_hem_h//2),(pad_hem_w//2+1,pad_hem_w//2)),mode='constant'),(e,f,g,h)
            # return np.pad(hem_bin,((pad_hem_h,0),(pad_hem_w//2+1,pad_hem_w//2)),mode='constant')
        else:
            e,f,g,h = pad_hem_h//2,pad_hem_h//2,pad_hem_w//2,pad_hem_w//2
            return np.pad(hem_bin,((pad_hem_h//2,pad_hem_h//2),(pad_hem_w//2,pad_hem_w//2)),mode='constant'),(e,f,g,h)

    def hem_unpad(self,mask,e,f,g,h):
        return mask[e:mask.shape[0]-f,g:mask.shape[1]-h]
        # return mask[pad_hem_h//2:mask.shape[0]-pad_hem_h//2,pad_hem_w//2:(mask.shape[1]-pad_hem_w//2)]#93
        # return mask[pad_hem_h//3+1:(mask.shape[0]-2*pad_hem_h//3),:(mask.shape[1]-pad_hem_w//2)]#94
