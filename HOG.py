# -*- coding: utf-8 -*-
from DB import Database

from skimage.feature import hog
from skimage import io

import numpy as np
import os

import matplotlib.pyplot as plt

#n_bin    = 128
#n_orient = 8
#p_p_c    = (32, 32)
#c_p_b    = (1, 1)
#depth    = 5

# cache dir
#cache_dir = 'cache'
#if not os.path.exists(cache_dir):
#    os.makedirs(cache_dir)


class HOG(object):
    def __init__(self):
        self.n_bin    = 128
        self.n_orient = 8
        self.p_p_c    = (32, 32)
        self.c_p_b    = (1, 1)
    
    def histogram(self, input_data, normalize=True):
        ''' count img histogram
        
          arguments
            input    : a path to a image or a numpy.ndarray
            n_bin    : number of bins of histogram
            type     : 'global' means count the histogram for whole image
                       'region' means count the histogram for regions in images, then concatanate all of them
            normalize: normalize output histogram
        
          return
            type == 'global'
              a numpy array with size n_bin
            type == 'region'
              a numpy array with size n_bin * n_slice * n_slice
        '''
        if isinstance(input_data, np.ndarray):  # examinate input type
            img = input_data.copy()
        else:
            img = io.imread(input_data, as_gray=False)
        hist = self.compute_HOG(img, self.n_bin)

        return hist

    def compute_HOG(self, img, n_bin, normalize=True):
        fd = hog(img, orientations=self.n_orient, pixels_per_cell=self.p_p_c, cells_per_block=self.c_p_b,
               feature_vector=True, block_norm='L2-Hys', visualize=False, multichannel=True)
        hist, _ = np.histogram(fd, bins=n_bin)

        if normalize:
            hist = np.array(hist) / np.sum(hist)

        return hist
    
    
    def create_vector_db(self, db):
        print('creating fv db...')
#        with open('hog_fv_db.csv', 'w', encoding='UTF-8') as f:
#        f.write("img,cls,fv")
        data = db.get_data()
        size = len(db)
        fv_list = np.memmap(filename='hog_fv_data-cache.npy', mode='w+', shape=(size,4), dtype=np.object)
        for i, d in enumerate(data.itertuples(),1):
            print('\rprocess: {}/{}'.format(i,size), end='')
            img_loc = getattr(d, "img_loc")
            group_id = getattr(d, "group_id")
            img_id = getattr(d, "img_id")
            d_hist = self.histogram(img_loc)
            fv_list[i-1] = [img_loc, group_id, img_id, d_hist]
            #f.write("\n{},{},{}".format(d_img, d_cls, d_hist))
        print('\nfv db created')
        np.save('hog_fv_data.npy', fv_list)
        del fv_list
#        print(fv_list.shape)
#        return fv_list

    def distance(self, v1, v2):
        assert v1.shape == v2.shape, "shape of two vectors need to be same!"
        return np.sum(np.absolute(v1 - v2))
    
    def create_dis_list_img(self, img_name, size):
        hist = self.histogram(img_name)
        dis_list = []
        db_fv = np.memmap(filename='hog_fv_data.npy', mode='r', shape=(size, 4), dtype=np.object)
        for i, data in enumerate(db_fv, 1):
            print('\rprocess: {}/{}'.format(i,len(db_fv)), end='')
            dis = self.distance(hist, data[2])
            dis_list.append([dis, i-1])
        print('\ndistance list created.')
        return sorted(dis_list)
    
if __name__ == "__main__":
    db = Database()
    a = HOG()
    a.create_vector_db(db)

