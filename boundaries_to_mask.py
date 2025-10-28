import cv2
import pandas as pd
import numpy as np
import tifffile as ti
from tqdm import tqdm
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt
from joblib import Parallel, delayed
import multiprocessing
from scipy.ndimage import zoom
from time import time

# from parallel_processing import CellMaskBatchProcessor, resolve_overlap_fill


class BoundaryCSV:
    def __init__(self,csv,dapi,mpp=0.2125,**kwargs):
        self.csv = csv
        self.dapi = dapi
        self.mpp = mpp
        self.num_processes=multiprocessing.cpu_count()-1
        self.data = self.__read_csv(**kwargs)

    def render_cell(self,obj, data, mpp, cc):
        x_vertex_list = data.loc[data['cell_id'] == obj, 'vertex_x'].values
        y_vertex_list = data.loc[data['cell_id'] == obj, 'vertex_y'].values

        L = np.stack([
            (x_vertex_list / mpp).astype(int),
            (y_vertex_list / mpp).astype(int)
        ], axis=1)

        x1, x2 = L[:, 0].min(), L[:, 0].max()
        y1, y2 = L[:, 1].min(), L[:, 1].max()

        ctr = L.copy()
        ctr[:, 0] -= x1
        ctr[:, 1] -= y1

        mask_temp = np.zeros((y2 - y1, x2 - x1), dtype=np.uint16)
        cv2.fillPoly(mask_temp, [ctr], color=cc)

        return (cc, obj, y1, y2, x1, x2, mask_temp)

    def __map_slide(self,cc,data,uniqueObjects):

        L = []

        obj = uniqueObjects[cc]
        cc+=1

        # Get vertex lists
        x_vertex_list = list(data.loc[data['cell_id'] == obj, 'vertex_x'])
        y_vertex_list = list(data.loc[data['cell_id'] == obj, 'vertex_y'])

        # Create list of vertices
        for i in range(len(x_vertex_list)):
            idx = i
            L.append([int((1/self.mpp) * float(x_vertex_list[idx])),
                      int((1/self.mpp) * float(y_vertex_list[idx]))])

        ctr = np.array(L, dtype=np.int32)
        ctr_append = ctr.copy()

        # all_contours.append(ctr_append)

        x1 = min(ctr[:,0])
        x2 = max(ctr[:,0])

        y1 = min(ctr[:,1])
        y2 = max(ctr[:,1])


        ctr[:,0]=ctr[:,0]-x1
        ctr[:,1]=ctr[:,1]-y1

        mask_temp=np.zeros((y2-y1,x2-x1))
        cv2.fillPoly(mask_temp,[ctr], cc)

        assert mask_temp.shape == (y2 - y1, x2 - x1)

        # mask[y1:y2,x1:x2]=mask_temp

        ys, xs = np.nonzero(mask_temp)
        ys_global = ys + y1
        xs_global = xs + x1

        return ys_global, xs_global, cc


    def __map_cleanup(self,new_nucs):
        if len(new_nucs)>0:
            new_nucs=new_nucs[0]
        else:
            new_nucs = []
        return new_nucs


    def get_mask(self,head='cell_id',x_key = 'vertex_x', y_key = 'vertex_y',**kwargs):

        cc=0
        
        time1 =time()

        mask = self.__initialize_mask(x_key,y_key,**kwargs)
        

        data = self.__read_csv(**kwargs)

        uniqueObjects = sorted(list(set(data[head])))

        print(f'Mapping {len(uniqueObjects)} objects')

        all_contours = []

        unqObjList = []
        time2 = time()
        
        # print(f'Prep: {time2-time1}')

        # for obj in tqdm(uniqueObjects):
            
            
        groups = data.groupby(head)

        for cc, (obj, df_group) in tqdm(enumerate(groups, start=1)):
            unqObjList.append([cc, obj])
        
            # Get vertices as NumPy arrays (vectorized)
            x_vertex_list = df_group[x_key].to_numpy(dtype=np.float32)
            y_vertex_list = df_group[y_key].to_numpy(dtype=np.float32)
        
            ctr = np.stack([
                (x_vertex_list / self.mpp).astype(np.int32),
                (y_vertex_list / self.mpp).astype(np.int32)
            ], axis=1)
        
            all_contours.append(ctr)

 
            cc += 1

            # unqObjList.append([cc,obj])

            # L = []

            # # Get vertex lists
            # x_vertex_list = list(data.loc[data[head] == obj, x_key])
            # y_vertex_list = list(data.loc[data[head] == obj, y_key])

            # for i in range(len(x_vertex_list)):
            #     idx = i
            #     L.append([int((1/self.mpp) * float(x_vertex_list[idx])),
            #               int((1/self.mpp) * float(y_vertex_list[idx]))])

            # ctr = np.array(L, dtype=np.int32)
            # # ctr = ctr//2###
            # ctr_append = ctr.copy()

            # all_contours.append(ctr_append)
            
            time3=time()
            # print(f'Contour create: {time3-time2}')

           
            cv2.fillPoly(mask,[ctr],cc)
            time4 = time()
            # print(f'Populate: {time4-time3}')
            # if cc==10:
            #     raise SystemExit

        df = pd.DataFrame(unqObjList)
        df.columns = ['cell_no','cell_id']

        return mask,all_contours,df

    def get_vertex(self,obj,head='cell_id',x_key='vertex_x',y_key='vertex_y',**kwargs):


        x_vertex_list = list(self.data.loc[self.data[head] == obj, x_key])
        y_vertex_list = list(self.data.loc[self.data[head] == obj, y_key])

        L=[]

        # Create list of vertices
        for i in range(len(x_vertex_list)):
            idx = i
            L.append([int((1/self.mpp) * float(y_vertex_list[idx])),
                      int((1/self.mpp) * float(x_vertex_list[idx]))])

        ctr = np.array(L, dtype=np.int32)

        return ctr

    def check_bounds(self,ctr,mask):

        cy, cx = np.mean(ctr[:, 1]), np.mean(ctr[:, 0])
        cy, cx = int(round(cy)), int(round(cx))

        if 0 <= cy < mask.shape[1] and 0 <= cx < mask.shape[0]:
            return mask[cx, cy] > 0
        else:
            return False

    def __read_csv(self,**kwargs):
        return pd.read_csv(self.csv,**kwargs)

    def __initialize_mask(self,x_key,y_key,**kwargs):
        if self.dapi is None:
            csv = pd.read_csv(self.csv,**kwargs)
            max_x = int(round(max(csv[x_key])/self.mpp)+10)
            max_y = int(round(max(csv[y_key])/self.mpp)+10)

            return np.zeros((max_y,max_x),dtype=np.uint8)
        else:
            return np.zeros_like(ti.imread(self.dapi),dtype=np.uint8)

    def __read_ome(self):
        with ti.TiffFile(self.dapi) as tiff:

            data = np.array(tiff.series[0].pages[0].asarray())
        return data

    def __get_bbox(self,mask,idx):
        rows, cols = np.where(mask == idx)

        # Calculate the bounding box coordinates
        x_min, x_max = np.min(cols), np.max(cols)
        y_min, y_max = np.min(rows), np.max(rows)

        return (x_min,x_max,y_min,y_max)


    def __maximize_bbox(self,mask,x1,x2,x3,x4,y1,y2,y3,y4):

        if x1<x3:
            x = x1
        else:
            x = x3

        if x2 > x4:
            xx = x2
        else:
            xx = x4

        if y1 < y3:
            y = y1
        else:
            y = y3

        if y2 > y4:
            yy = y2
        else:
            yy = y4

        if x < 0:
            x=0
        if xx > mask.shape[1]:
            xx = mask.shape[1]

        if y < 0:
            y=0
        if yy > mask.shape[0]:
            yy = mask.shape[0]


        return x,xx,y,yy


# csvs = "/orange/pinaki.sarder/nlucarelli/Xenium/HD and Xenium for Nick/Xenium/output-XETG00126__0010200__f59__20240214__210015/nucleus_boundaries.csv.gz"
# dapi = "/orange/pinaki.sarder/nlucarelli/Xenium/HD and Xenium for Nick/Xenium/output-XETG00126__0010200__f59__20240214__210015/morphology_focus.ome.tif"

# obj = BoundaryCSV(csvs,dapi)

# mask = obj.get_mask()


# ti.imwrite("/orange/pinaki.sarder/nlucarelli/Xenium/HD and Xenium for Nick/Xenium/output-XETG00126__0010200__f59__20240214__210015/seg_test.tif",mask,photometric='minisblack')
