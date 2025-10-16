#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:26:32 2024

@author: nlucarelli
"""
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from PAS_deconvolution import deconvolution,deconvolution_WSI
from skimage.filters import threshold_local,threshold_isodata, threshold_niblack
from skimage.morphology import remove_small_objects, binary_opening, binary_closing,disk
from tqdm import tqdm
from scipy import ndimage

class HE:
    def __init__(self,he,maxPatchWidth=2000):
        self.he = he
        self.maxPatchWidth = maxPatchWidth
        self.num_splits, self.num_workers = self.__get_splits()

        MODx=np.zeros((3,))
        MODy=np.zeros((3,))
        MODz=np.zeros((3,))
        MODx[0]= 0.644211
        MODy[0]= 0.716556
        MODz[0]= 0.266844
        # MODx[0]= 0.564
        # MODy[0]= 0.723
        # MODz[0]= 0.399


        MODx[1]= 0.175411
        MODy[1]= 0.972178
        MODz[1]= 0.154589

        # MODx[1]= 0.236
        # MODy[1]= 0.882
        # MODz[1]= 0.408

        MODx[2]= 0.0
        MODy[2]= 0.0
        MODz[2]= 0.0

        # MODx[2]= -0.159
        # MODy[2]= -0.379
        # MODz[2]= 0.911


        MOD=[MODx,MODy,MODz]
        self.MOD = MOD

    def segment_he(self,min_object_size=25):
        return self._process_image(min_object_size)

    def deconvolve(self):
        return self._process_image_d()


    def _split_image(self,image, num_splits):
        # Split the image into a grid of sub-images

        if len(image.shape) > 2:
            height,width,channel = image.shape
        else:
            height, width = image.shape

        split_height = height // num_splits
        split_width = width // num_splits


        crops = []
        xs = []
        ys = []

        for i in range(num_splits):
            for j in range(num_splits):
                if i==num_splits-1:
                    h_end = height
                else:
                    h_end = ((i+1)*split_height)

                if j==num_splits-1:
                    w_end = width
                else:
                    w_end = ((j+1)*split_width)

                if len(image.shape) > 2:
                    crop = image[i*split_height:h_end, j*split_width:w_end,:]
                else:
                    crop = image[i*split_height:h_end, j*split_width:w_end]
                crops.append(crop)
                xs.append(i*split_height)
                ys.append(j*split_width)
        return crops,xs,ys,(split_height,split_width)

    def _process_image(self,min_object_size):

        he_crops,xs,ys,splits = self._split_image(self.he,self.num_splits)

        crop_dicts = Parallel(n_jobs=int(self.num_workers**2))(delayed(self._process_crop)(he_crops[i],xs[i],ys[i]) for i in tqdm(range(len(he_crops)),desc='Processing crops....'))

        reconstructed = self._post_process(crop_dicts, splits)

        return reconstructed

    def _process_image_d(self):
        he_crops,xs,ys,splits = self._split_image(self.he,self.num_splits)
        crop_dicts = Parallel(n_jobs=int(self.num_workers**2))(delayed(self._process_crop_d)(he_crops[i],xs[i],ys[i]) for i in tqdm(range(len(he_crops)),desc='Processing crops....'))

        reconstructed = self._post_process(crop_dicts, splits)

        return reconstructed

    def _process_crop_d(self,he_crop,x,y):
        img_stain1,_,_ = deconvolution(he_crop)
        img_stain1=np.invert(img_stain1.astype('uint8'))
        crop_dict = {'crop':img_stain1,'x':x,'y':y}

        return crop_dict

    def _process_crop(self,he_crop,x,y):


        img_stain1,_,_ = deconvolution(he_crop)
        img_stain1=np.invert(img_stain1.astype('uint8'))
        hem_bin=(img_stain1>threshold_local(img_stain1,block_size=51,offset=-10)).astype('uint8')#-30
        hem_bin = hem_bin.astype('bool')


        # labeled, num_features = ndimage.label(hem_bin)

        # # Get object IDs that touch the border
        # border_ids = set(np.unique(labeled[0, :])) | set(np.unique(labeled[-1, :])) | \
        #              set(np.unique(labeled[:, 0])) | set(np.unique(labeled[:, -1]))
        # border_ids.discard(0)  # Remove background
        # edge_mask = np.isin(labeled, list(border_ids))



        # reconstructed=remove_small_objects(hem_bin,25)
        # hem_bin=binary_closing(hem_bin,disk(5))
        # # reconstructed=remove_small_objects(hem_bin,25)
        # hem_bin = (hem_bin | edge_mask)

        hem_bin = 255*hem_bin.astype(np.uint8)

        crop_dict = {'crop':hem_bin,'x':x,'y':y}

        return crop_dict

    def _post_process(self,crop_dicts,splits):


        split_x,split_y = splits
        x,y,_ = self.he.shape
        reconstructed = np.uint8(np.zeros((x,y)))


        for i in tqdm(range(len(crop_dicts)),desc='Post processing...'):

            x_i = crop_dicts[i]['x']
            y_i = crop_dicts[i]['y']


            x_end = x if ((x-(x_i + split_x)) < split_x) else (x_i + split_x)
            y_end = y if ((y-(y_i + split_y)) < split_y) else (y_i + split_y)

            # try:

            reconstructed[x_i:x_end,y_i:y_end] = crop_dicts[i]['crop']
            # except:
            #     print(f'Oh Naur!! Failed at {i}')
            #     print(x_end-x_i)
            #     print(split_x)
            #     print(crop_dicts[i]['crop'].shape)

        reconstructed=remove_small_objects(reconstructed,50)
        reconstructed=binary_closing(reconstructed,disk(3))
        reconstructed=remove_small_objects(reconstructed,50)

        return reconstructed
    def __get_splits(self):
        worker_count = int(np.floor(multiprocessing.cpu_count()**0.5))

        height,width = self.he.shape[:2]

        print(self.he.shape)
        print(height)
        print(width)

        split_height = height // worker_count
        split_width = width // worker_count

        if split_height > self.maxPatchWidth:
            split_height = self.maxPatchWidth

        if split_width > self.maxPatchWidth:
            split_width = self.maxPatchWidth

        splits_h = np.floor(height / split_height)
        splits_w = np.floor(width / split_width)

        splits = splits_h if splits_h > splits_w else splits_w


        return int(splits), worker_count



import tiffslide as openslide
import numpy as np
import matplotlib.pyplot as plt
import tifffile as ti
import time
from glob import glob

#23568
filenames = glob("/orange/pinaki.sarder/nlucarelli/Xenium/R01_2/*converted.tif")
output_dir = "/orange/pinaki.sarder/nlucarelli/Xenium/R01_2/"

for filename in filenames:

    sl = openslide.OpenSlide(filename)
    x,y = sl.dimensions

    im = sl.read_region((0,0),0,(x,y))
    im = np.array(im)[:,:,:3]

    he = HE(im)

    time0 = time.time()
    segmentation = he.segment_he(25)
    # deconv = he.deconvolve()
    time1 = time.time()


    print(f'Segmentation Time: {time1-time0}')
    outnm = output_dir + filename.split('.tif')[0].split('/')[-1]+'_hem.tif'
    print(f'Saving {outnm}...')


    ti.imwrite(outnm,segmentation,photometric='minisblack')
