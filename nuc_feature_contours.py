#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:37:41 2025

@author: nlucarelli
"""
import numpy as np
import tifffile as ti
import tiffslide as openslide
import pandas as pd
import cv2
# import concurrent.futures
from joblib import Parallel,delayed
from multiprocessing import Pool
from skimage.measure import regionprops
from skimage.morphology import binary_dilation,disk
from skimage.color import rgb2gray
from skimage.feature import graycomatrix,graycoprops
from tqdm import tqdm
from time import time
import warnings
warnings.filterwarnings("ignore")

class Paired_Data():
    def __init__(self,he_filename,csv,tf_mat,mask_shape,filter_csv=None):
        self.he_filename = he_filename
        self.data = pd.read_csv(csv)
        
        self.filter_csv = filter_csv
        
        if self.filter_csv is not None:
            self.filter = pd.read_csv(self.filter_csv)
            
            filtered = sorted(list(np.unique(self.filter['cell_id'])))

            self.data = self.data.loc[self.data['cell_id'].isin(filtered),['cell_id','vertex_x','vertex_y','label_id']]
            
        else:
            self.filter = None
            
        self.uniqueObjects = sorted(list(set(self.data['cell_id'])))
        self.tf_mat = tf_mat
        self.mpp = 0.2125
        self.mask_shape = mask_shape
        self.center_coord_y = mask_shape[0]//2
        self.center_coord_x = mask_shape[1]//2
        
        he = openslide.OpenSlide(self.he_filename)

        dimensions = he.dimensions
        
        self.he_im = np.array(he.read_region((0,0),0,dimensions))
        
        self.contour_dict = {
            cell_id: group[['vertex_x', 'vertex_y']].values
            for cell_id, group in self.data.groupby('cell_id')
                }
        

    def _transform_contours(self,contours):

        # rot=3
        # flip=0
        # exp_factor = 0.47
        tf_mat = np.rot90(self.tf_mat[0:2,:],k=2)
        
        rot = self.tf_mat[3,1]
        flip = self.tf_mat[3,0]
        exp_factor = self.tf_mat[4,0]
        
        offset = self.tf_mat[2,:]
        
        
        # pad_dap = self.tf_mat[3,:]
        # pad_hem = self.tf_mat[4,:]
        
        pad_dap = self.tf_mat[5:7,:]
        pad_hem = self.tf_mat[7:9,:]
        
        small_size = self.tf_mat[9,0]
        

        transformed_contours = []
                
        
        if small_size:
            offset=2*self.tf_mat[2,:]
            # exp_factor=2*self.tf_mat[4,0]

        for contour in contours:
            coords = np.copy(contour).astype(np.float32)  # shape (N, 2), [row, col] = [y, x]


            # Compute centroid of the contour (in [y, x])
            M = cv2.moments(coords)
            if M['m00'] == 0:
                transformed_contours.append(coords)
                continue


            cy = M['m10'] / M['m00']  # row
            cx = M['m01'] / M['m00']  # col


            # Shift to origin around centroid
            coords -= [cy, cx]

        

            # coords_og = np.copy(coords)

            # Apply zoom
            # coords *= exp_factor

            # Rotate 90° counter-clockwise (rot=1)
            angle = 90 * rot
            rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)

            # Apply rotation manually (since OpenCV expects [x, y])
            coords_rot = np.zeros_like(coords)
            coords_rot[:, 1] = rotation_matrix[0, 0] * coords[:, 1] + rotation_matrix[0, 1] * coords[:, 0]
            coords_rot[:, 0] = rotation_matrix[1, 0] * coords[:, 1] + rotation_matrix[1, 1] * coords[:, 0]

            # Apply flip
            if flip == 1:
                coords_rot[:, 0] = -coords_rot[:, 0]  # vertical flip (y)
            elif flip == 2:
                coords_rot[:, 1] = -coords_rot[:, 1]  # horizontal flip (x)

            # Reposition relative to image center

            # Translate back to original center

            

            cy2 = cx*exp_factor
            cx2 = cy*exp_factor


            coords_shifted = np.copy(coords_rot)


            coords_shifted *= exp_factor
            
            addition_x = self.mask_shape[0]*exp_factor#CHECK THIS SECTION TO MAKE SURE EVERYTHING IS CORRECT
            addition_y = self.mask_shape[1]*exp_factor
            
            
            if ((flip==2) & (rot==1)):
                
                coords_shifted = coords_shifted + [addition_y-cy2, addition_x-cx2]
                
            else:
                coords_shifted = coords_shifted + [cy2, addition_x-cx2]
                
                # print(f'Something isnt right, check out {self.he_filename}')
                
            
            

            # Padding before affine
            if pad_dap is not None:
                a = pad_dap[0,0]
                c = pad_dap[1,0]
                
                if small_size:
                    a=2*pad_dap[0,0]
                    c=2*pad_dap[1,0]
                    
                # pad_h, pad_w = pad_dap
                coords_shifted[:, 0] += a#pad_h//2
                coords_shifted[:, 1] += c#pad_w//2
                

            coords_shifted = coords_shifted[:,::-1]

            tf_inv = np.linalg.inv(tf_mat)

            # coords_shifted = (coords_shifted) @ tf_inv - np.flip(offset)
            coords_shifted = (coords_shifted-np.flip(offset)) @ tf_inv.T

            coords_shifted = coords_shifted[:,::-1]
            

            # Remove padding after affine
            if pad_hem is not None:
                # pad_h, pad_w = pad_hem
                e = pad_hem[0,0]
                g = pad_hem[1,0]
                
                if small_size:
                    e=2*pad_hem[0,0]
                    g=2*pad_hem[1,0]
                
                # coords_shifted[:, 0] -= pad_h // 2
                coords_shifted[:,0] -= e#pad_h//2
                coords_shifted[:, 1] -= g#pad_w // 2
                
            # if small_size:
            #     coords_shifted*=2

            transformed_contours.append(coords_shifted)
                        

            return transformed_contours[0]
    def _get_bbox(self,L):
        x1, x2 = L[:, 0].min(), L[:, 0].max()
        y1, y2 = L[:, 1].min(), L[:, 1].max()

        x1,x2 = int(np.floor(x1)),int(np.ceil(x2))
        y1,y2 = int(np.floor(y1)),int(np.ceil(y2))

        return (x1,x2,y1,y2)
    
    def _get_centroid(self,L):
        
        L = np.array(L,dtype=np.int32)
        
        M = cv2.moments(L)
        cy = M['m10'] / M['m00']
        cx = M['m01'] / M['m00']
        
        return cy,cx
        

    def _center_and_display(self,L,he_im=None,return_intensity=False):
        x1, x2 = L[:, 0].min(), L[:, 0].max()
        y1, y2 = L[:, 1].min(), L[:, 1].max()

        ctr = L.copy()
        ctr[:, 0] -= x1
        ctr[:, 1] -= y1

        x1,x2 = int(np.floor(x1)),int(np.ceil(x2))
        y1,y2 = int(np.floor(y1)),int(np.ceil(y2))

        sizex = int(x2-x1)
        sizey = int(y2-y1)


        mask_temp = np.zeros((sizex,sizey), dtype=np.uint8)
        ctr = np.int32(np.round(ctr))
        cv2.fillPoly(mask_temp, [ctr[:,::-1]], color=1)
        

        # mask_temp = np.rot90(mask_temp,k=-1)

        if return_intensity:
            # he = openslide.OpenSlide(self.he_filename)
            pad = 0
            he_temp = he_im[x1-pad:x2+pad,y1-pad:y2+pad,:]
            # he_temp = he.read_region((y1,x1),0,(y2-y1,x2-x1))
            # he_temp = he.read_region((y1,x1),0,(y2-y1,x2-x1))
            # he_temp = np.array(he_temp)
            ctr[:,1]+=(pad)#Left Right
            ctr[:,0]+=(pad)#Up Down
            
            # cv2.drawContours(he_temp, [ctr[:,::-1]], -1, (255,0,0), 1)

            return mask_temp,he_temp
        else:

            return mask_temp


    def _get_contour(self,idx):
        
        obj = self.uniqueObjects[idx]#12!
        
        
        contour = self.contour_dict[obj]
        x_vertex_list = contour[:,0]
        y_vertex_list = contour[:,1]
        
        # x_vertex_list = list(self.data.loc[self.data['cell_id'] == obj, 'vertex_x'])
        # y_vertex_list = list(self.data.loc[self.data['cell_id'] == obj, 'vertex_y'])

        L=[]
        for i in range(len(x_vertex_list)):
                idx = i
                L.append([int((1/self.mpp) * float(y_vertex_list[idx])),
                          int((1/self.mpp) * float(x_vertex_list[idx]))])

        ctr = [np.array(L, dtype=np.int32)]
        
        tf_ctr = self._transform_contours(ctr)
        
    
        return tf_ctr


    def test_contour(self,obj,cyto=None):
        # uniqueObjects = sorted(list(set(self.data['cell_id'])))
                
        contour = self.contour_dict[obj]
        x_vertex_list = contour[:,0]
        y_vertex_list = contour[:,1]

        # obj = uniqueObjects[0]#12!
        # obj = 'aaaljiig-1'
        # print(obj)
        # x_vertex_list = list(self.data.loc[self.data['cell_id'] == obj, 'vertex_x'])
        # y_vertex_list = list(self.data.loc[self.data['cell_id'] == obj, 'vertex_y'])

        L=[]
        for i in range(len(x_vertex_list)):
                idx = i
                L.append([int((1/self.mpp) * float(y_vertex_list[idx])),
                          int((1/self.mpp) * float(x_vertex_list[idx]))])

        ctr = [np.array(L, dtype=np.int32)]
                        
        tf_ctr = self._transform_contours(ctr)
        
        # tf_ctr = self._get_contour(obj)
        
        # cntr = self._get_centroid(tf_ctr)
                
        # return tf_ctr
        
        # he = openslide.OpenSlide(self.he_filename)

        # dimensions = he.dimensions
        
        # he_im = np.array(he.read_region((0,0),0,dimensions))
                
        mask,he2 = self._center_and_display(tf_ctr,he_im=self.he_im,return_intensity=True)
        # mask2,he2 = self.__center_and_display(og)
        
        if cyto is not None:
            bbox = self._get_bbox(tf_ctr)
            x1,x2,y1,y2 = bbox

            ins = cyto[x1:x2,y1:y2]
            ins = ins>0
            mask = np.uint8((mask>0) &  ~ins)


        return mask,he2
    
    def map_structure(self,im,bbox,ids,mapping_dict):
        
        for i in range(len(ids)):
            
            st = ids.iloc[i,0]
            gr = ids.iloc[i,1]
            
            L = self.test_contour(st)
            
            ctr = L.copy()
            ctr[:, 0] -= bbox[1]
            ctr[:, 1] -= bbox[0]
            
            ctr = np.int32(np.round(ctr))
            
            # print(ctr)
            # print(st)
            # print(gr)
            # raise SystemExit
            
            # ctr[:,1]+=(30)
            # ctr[:,0]+=(-1)
            
            cv2.drawContours(im, [ctr[:,::-1]], -1, mapping_dict[gr], 1)
        
        
        return im


    def _clean_lists(self,lists):

        lists = [np.array(list_temp) for list_temp in lists]
        lists = [np.expand_dims(list_temp,-1) if len(list_temp.shape)==1 else list_temp for list_temp in lists]
        lists = np.concatenate(lists,axis=1)

        return lists


    def _get_max_nuclei(self):
        return len(self.uniqueObjects)

    def _correct_bounding_box(self,bb,radius):
        bound_x,bound_y = self.dapi.shape

        min_x = bb[0]
        min_y = bb[1]
        max_x = bb[2]
        max_y = bb[3]

        if (min_x-radius) < 0:
            min_x = radius
        if (min_y-radius) < 0:
            min_y = radius
        if (max_x+radius) > bound_x:
            max_x = bound_x-radius
        if (max_y+radius) > bound_y:
            max_y = bound_y - radius


        return (min_x,min_y,max_x,max_y)

    def _compute_texture_features(self,args):
        idx, contour, dilate, radius = args

        try:
            mask,he = self._center_and_display(contour, return_intensity=True)
            props = regionprops(mask)

            if len(props) != 1:
                return None

            p = props[0]
            # bbox = p.bbox
            window = np.copy(mask)
            he_window = np.copy(he)
            label_int = p.label
            binary = window == label_int

            bw = rgb2gray(he_window)
            bw = (255 * bw).astype(np.int16)
            bw[~binary] = 256  # background level

            glcm = graycomatrix(bw, [1, 2, 3], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=257, symmetric=True, normed=True)
            glcm = glcm[:256, :256, :, :]  # crop background

            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]

            return [idx, contrast, dissimilarity, homogeneity, energy, correlation]

        except Exception as e:
            print(f"[Index {idx}] Texture error: {e}")


    def get_texture_features(self,dilate=False,radius=3):

        max_nuclei = self._get_max_nuclei()

        contours = [self._get_contour(i) for i in tqdm(range(max_nuclei), desc='Getting contours...')]

        args = [(i, ctr, dilate, radius) for i, ctr in enumerate(contours)]

        with Pool() as pool:
            results = list(tqdm(pool.imap(self._compute_texture_features, args), total=len(args), desc='Extracting Texture Features...'))

        results = [r for r in results if r is not None]

        data = self._clean_lists(list(map(list, zip(*results))))# transpose
        data = pd.DataFrame(data)
        data.columns = ['idx', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

        return data



    def _compute_shape_features(self,args):

        index, contour = args

        try:
            mask = self._center_and_display(contour,False)

            props = regionprops(mask)

            if len(props) != 1:
                return None

            p = props[0]
            aspect_ratio = p.axis_major_length / p.axis_minor_length if p.axis_minor_length > 0 else 0
            shape_factor = p.perimeter**2 / p.area if p.perimeter > 0 else 0

            return [
                index,
                p.area,
                aspect_ratio,
                p.solidity,
                p.eccentricity,
                *p.moments_hu,  # flatten the 7 moments
                shape_factor
            ]
        except Exception as e:
            print(f'[Index {index}] Error: {e}')
            return None


    def get_shape_features(self):

        max_nuclei = self._get_max_nuclei()

        contours = [self._get_contour(i) for i in tqdm(range(max_nuclei),desc='Getting contours')]
        args = [(i,ctr) for i,ctr in enumerate(contours)]


        with Pool() as pool:
            results = list(tqdm(pool.imap(self._compute_shape_features,args),total=len(args),desc='Gettin shape features...'))

        results = [r for r in results if r is not None]

        data = self._clean_lists(list(zip(*results)))
        data = pd.DataFrame(data)
        data.columns = ['idx','Area','Aspect_Ratio','Solidity','Eccentricity','Hu_Moment_1','Hu_Moment_2','Hu_Moment_3',
                        'Hu_Moment_4','Hu_Moment_5','Hu_Moment_6','Hu_Moment_7','Shape_Factor']

        return data

    def _compute_intensity_features(self,args):
        index,contour, radius, dilate = args

        try:
            mask, intensity_image = self._center_and_display(contour, return_intensity=True)

            props = regionprops(mask, intensity_image=intensity_image)
            if len(props) != 1:
                return None  # Skip or handle error

            p = props[0]
            mins = p.intensity_min
            maxs = p.intensity_max
            means = p.intensity_mean
            stds = np.std((p.image_intensity[p.image])[:, :3], axis=0)  # Assumes RGB

            return [index,
                mins[0], maxs[0], means[0], stds[0],
                mins[1], maxs[1], means[1], stds[1],
                mins[2], maxs[2], means[2], stds[2],
            ]

        except Exception as e:
            print(f'[Index {index}] Error: {e}')
            return None





    def get_intensity_features(self, dilate=False, radius=3):
        max_nuclei = self._get_max_nuclei()

        # Precompute contours so OpenSlide object is not passed to workers
        contours = [self._get_contour(i) for i in tqdm(range(max_nuclei),desc='Getting contours...')]
        args = [(i,ctr, radius, dilate) for i,ctr in enumerate(contours)]

        # Partial application: closure to keep self._center_and_display accessible

        with Pool() as pool:
            results = list(tqdm(pool.imap(self._compute_intensity_features, args), total=len(args),desc='Getting intensity features...'))

        # Remove None results (if any)
        results = [r for r in results if r is not None]

        data = self._clean_lists(list(map(list, zip(*results))))  # transpose
        data = pd.DataFrame(data)
        data.columns = ['idx',
            'red_min', 'red_max', 'red_mean', 'red_std',
            'green_min', 'green_max', 'green_mean', 'green_std',
            'blue_min', 'blue_max', 'blue_mean', 'blue_std'
        ]

        return data

    def get_label_features(self):
        max_nuclei = self._get_max_nuclei()

        labels = []

        for nuc in tqdm(range(max_nuclei)):
            labels.append(self.uniqueObjects[nuc])

        data = self._clean_lists([labels])
        data = pd.DataFrame(data)
        data.columns = ['cell_id']

        return data


    def _compute_all_features(self,args,he_im=None):

        index, contour,cyto = args
                
        # time1 = time()
        
        cent_x,cent_y = self._get_centroid(contour)
        
        # time2 = time()
        
        try:
            mask,ii = self._center_and_display(contour,he_im=he_im,return_intensity=True)
            
            # time3 = time()
            

            props = regionprops(mask,intensity_image=ii)

            if len(props) != 1:
                return None

            p = props[0]
            
            if any([x==0 for x in ii.shape]):
                aspect_ratio = p.axis_major_length / p.axis_minor_length if p.axis_minor_length > 0 else 0
                shape_factor = p.perimeter**2 / p.area if p.perimeter > 0 else 0
                return [
                    index,cent_x,cent_y,
                    0,0,0,0,
                    0,0,0,0,
                    0,0,0,0,
                    p.area,
                    aspect_ratio,
                    p.solidity,
                    p.eccentricity,
                    *p.moments_hu,  # flatten the 7 moments
                    shape_factor,
                    0,
                    0,
                    0,
                    0,
                    0,
                ] 

            mins = p.intensity_min
            maxs = p.intensity_max
            means = p.intensity_mean
            stds = np.std((p.image_intensity[p.image])[:, :3], axis=0)

            if cyto is not None:
                bbox = self._get_bbox(contour)
                x1,x2,y1,y2 = bbox

                ins = cyto[x1:x2,y1:y2]
                ins = ins>0
                mask = np.uint8((mask>0) &  ~ins)
                props = regionprops(mask,intensity_image=ii)


            aspect_ratio = p.axis_major_length / p.axis_minor_length if p.axis_minor_length > 0 else 0
            shape_factor = p.perimeter**2 / p.area if p.perimeter > 0 else 0
            
            
            window = np.copy(mask)
            he_window = np.copy(ii)
            label_int = p.label
            binary = window == label_int

            bw = rgb2gray(he_window)
            bw = (255 * bw).astype(np.int16)
            bw[~binary] = 256  # background level

            glcm = graycomatrix(bw, [1, 2, 3], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=257, symmetric=True, normed=True)
            glcm = glcm[:256, :256, :, :]  # crop background

            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            
            
            
            # time4 = time()
            
            # print(f'Centroid Time: {time2-time1}')
            # print(f'Center Time: {time3-time2}')
            # print(f'Feat Time: {time4-time3}')
            # raise SystemExit
            
              # Assumes RGB

            return [
                index,cent_x,cent_y,
                mins[0], maxs[0], means[0], stds[0],
                mins[1], maxs[1], means[1], stds[1],
                mins[2], maxs[2], means[2], stds[2],
                p.area,
                aspect_ratio,
                p.solidity,
                p.eccentricity,
                *p.moments_hu,  # flatten the 7 moments
                shape_factor,
                contrast,
                dissimilarity,
                homogeneity,
                energy,
                correlation,
            ]

        except Exception as e:
            # mask,ii = self._center_and_display(contour,he_im=he_im,return_intensity=True)
            print(f'[Index: {index}] Error: {e}')
            
    def get_all_contours(self):


        max_nuclei = self._get_max_nuclei()

        # cells_oi = ["EC-GC","I-EC","pEC"]
        # ex = pd.read_csv("/orange/pinaki.sarder/nlucarelli/Xenium/Xenium_HE/3775/3775_anchors_noECNS.csv")
        # ids = ex.loc[ex['group'].isin(cells_oi),'cell_id']  
        
        # he = openslide.OpenSlide(self.he_filename)

        # dimensions = he.dimensions
        
        # he_im = np.array(he.read_region((0,0),0,dimensions))
        

        contours = [self._get_contour(i) for i in tqdm(range(max_nuclei),desc='Getting contours...')]
        
        return contours


    def get_all_features(self,cyto=None,types=None):


        max_nuclei = self._get_max_nuclei()

        # cells_oi = ["EC-GC","I-EC","pEC"]
        # ex = pd.read_csv("/orange/pinaki.sarder/nlucarelli/Xenium/Xenium_HE/3775/3775_anchors_noECNS.csv")
        # ids = ex.loc[ex['group'].isin(cells_oi),'cell_id']  
        
        he = openslide.OpenSlide(self.he_filename)

        dimensions = he.dimensions
        
        he_im = np.array(he.read_region((0,0),0,dimensions))
        

        contours = [self._get_contour(i) for i in tqdm(range(max_nuclei),desc='Getting contours...')]
        
        # contours = [self._get_contour(i) for i in tqdm(self.uniqueObjects,desc='Getting contours...')]
        args = [(self.uniqueObjects[i],ctr,cyto) for i,ctr in enumerate(contours)]

        # with Pool() as pool:
            # results = list(tqdm(pool.imap(self._compute_all_features,args),total=len(args),desc='Getting all features...'))

        results = Parallel(n_jobs=1)(delayed(self._compute_all_features)(arg,he_im) for arg in tqdm(args,desc='features...'))

        results = [r for r in results if r is not None]

        data = self._clean_lists(list(map(list,zip(*results))))
        data = pd.DataFrame(data)
        data.columns = ['cell_id','centroid_x','centroid_y',
            'red_min', 'red_max', 'red_mean', 'red_std',
            'green_min', 'green_max', 'green_mean', 'green_std',
            'blue_min', 'blue_max', 'blue_mean', 'blue_std',
            'Area','Aspect_Ratio','Solidity','Eccentricity','Hu_Moment_1','Hu_Moment_2','Hu_Moment_3',
            'Hu_Moment_4','Hu_Moment_5','Hu_Moment_6','Hu_Moment_7','Shape_Factor',
            'contrast','dissimilarity','homogeneity','energy','correlation',
            ]

        
        if types is not None:
            cell_types = types
            
            data = data.merge(cell_types[['cell_id', 'group_x','group_y']], on='cell_id', how='inner')
            

        return data
