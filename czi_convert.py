#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:18:53 2023

@author: nlucarelli
"""
from czifile import CziFile
import numpy as np
import tifffile as ti
# import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from tqdm import tqdm


def disp_np(a):

    im = Image.fromarray(a)
    im.show()

def save_pyramid(data,filename):

    with ti.TiffWriter(filename,bigtiff=True) as tif:

        x,y,_ = data.shape
        max_dim = x if x > y else y


        thumbnail_downsample = (max_dim // 1024) + 1

        levels = (sum([np.log2(x).is_integer() for x in range(2,thumbnail_downsample)]))//2+1


        # metadata = {
        #     'Pyramid':'true',
        # }

        desc_str = 'Aperio Fake|AppMag = 40|MPP=0.25'

        options = {
            'tile': (240, 240),  # Tile size for efficient access
            'compression': 'jpeg',  # Optional compression
            'photometric': 'rgb',  # For color images
        }


        tif.write(data,
                  description=desc_str,
                  subfiletype=0,
                  metadata=None,
                  resolution=(0.25, 0.25),
                  **options)

        print(f'Level 0: {data.shape}')

        thumbnail = data[::thumbnail_downsample,::thumbnail_downsample,:]
        print(f'Thumbnail: {thumbnail.shape}')
        tif.write(thumbnail,
                  subfiletype=1,
                  description='Thumbnail',
                  metadata={'Name': 'Thumbnail'})


        # Write the pyramid levels
        for level in range(1, levels):  # Create 3 pyramid levels
            subsampled_data = data[::4**(level), ::4**(level),:]
            print(f'Level {level}: {subsampled_data.shape}')
            resolution_x = 0.25 * (4 ** level)
            resolution_y = 0.25 * (4 ** level)
            tif.write(subsampled_data,
                        # subfiletype=1,
                      description='Level '+str(level),
                      resolution=(resolution_x, resolution_y),
                      metadata={'Name':'Level '+str(level)},
                      **options)




downsample = 1.0

slide_dir = "/blue/sarder-hubmap/nlucarelli/jamie/"
ext = '.tif'

output_dir = "/blue/sarder-hubmap/nlucarelli/jamie/converted/"


for filename in tqdm(glob(slide_dir+'*'+ext),desc='Converting...'):

    basename = filename.split(ext)[0]
    print('Working on: ' + filename.split('/')[-1])
    # im = CziFile(filename)
    # thumbnail = np.squeeze(im.asarray())

    thumbnail = ti.imread(filename)
    
    assert len(thumbnail.shape)==3
    
    if thumbnail.shape[0]==3:
        thumbnail = np.transpose(thumbnail,[1,2,0])

    # save_pyramid(thumbnail,basename+'_converted.tif')
    save_pyramid(thumbnail, output_dir+filename.split(ext)[0].split('/')[-1]+'.tif')
