from HOG import HOG
from DB import Database
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

import os

db = Database()
a = HOG()
#root_dir = 'query_images'
#img_files = os.listdir(root_dir)
#print(img_files)
#
#results = []
#n = 3
#for img_name in img_files:
#    d = a.create_dis_list(os.path.join(root_dir,img_name))
#    results.append(d[:n])
#
#print(len(results))

size=len(db)
db_fv = np.load('hog_fv_data.npy')

dis_list = []
for i in range(size):
    first = db_fv[i]
    print('\rprocess: {}/{}'.format(i+1,size), end='')
    min_dis = 9999999.9
    index = -1
    for j in range(size):
        if i == j:
            continue
        second = db_fv[j]
        dis = a.distance(second[-1], first[-1])
        if min_dis > dis:
            min_dis = dis
            index = j
    dis_list.append([i, min_dis, j])
print('\ndistance list created.')

acc = 0
for i in range(size):
    x_idx = dis_list[i][0]
    y_idx = dis_list[i][-1]
    if db_fv[y_idx][1] == db_fv[x_idx][1]:
        acc += 1
    print('\rprocess: {}/{}'.format(i+1,size), end='')
    
print('\naccuracy: {:.3f}'.format(acc/size))
