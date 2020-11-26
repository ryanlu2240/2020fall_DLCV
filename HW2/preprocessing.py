import numpy as np
import cv2
import os
import pandas as pd
import h5py
from shutil import copyfile
from sklearn.model_selection import train_test_split
from PIL import Image

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs


tmp_dir = 'tmp_data/train/'

f = h5py.File('digitStruct.mat','r')
index = list(range(f['/digitStruct/bbox'].shape[0]))
x_idx, y_idx, y_train, y_test = train_test_split(index, index, test_size=0.2, shuffle=False)


for j in range(f['/digitStruct/bbox'].shape[0]):
    print(j)
    img_name = get_name(j, f)
    row_dict = get_bbox(j, f)
    ann = []
    src = tmp_dir + img_name
    im = cv2.imread(src)
    height = im.shape[0]
    width = im.shape[1]
    for it in range(len(row_dict['label'])):
        x1 = row_dict['left'][it]
        y1 = row_dict['top'][it]
        x2 = x1 + row_dict['width'][it]
        y2 = y1 + row_dict['height'][it]
        id = row_dict['label'][it]
        if id == 10.0:
            id=0.0
        cen_x = (x1 + x2) / 2
        cen_y = (y1 + y2) / 2
        box_width = x2 - x1
        box_height = y2 - y1
        cen_x = abs(cen_x / width)
        cen_y = abs(cen_y / height)
        box_width = abs(box_width / width)
        box_height = abs(box_height / height)
        id = int(id)
        t = str(id) + ' ' + str(cen_x) + ' ' + str(cen_y) + ' ' + str(box_width)  + ' ' + str(box_height)
        ann.append(t)
    if j in x_idx:
        fn = 'data/obj/' + img_name[:-4] + '.txt'
        train_f = open(fn, "w")
        for id in ann:
            if id != ann[-1]:
                id = id + '\n'
            train_f.write(id)
        train_f.close()

        # src = tmp_dir + img_name
        # dst = 'data/obj/' + img_name[:-4] + '.jpg'
        # im = Image.open(src)
        # rgb_im = im.convert('RGB')
        # rgb_im.save(dst)

    if j in y_idx:
        fn = 'data/test/' + img_name[:-4] + '.txt'
        test_f = open(fn, "w")
        for id in ann:
            if id != ann[-1]:
                id = id + '\n'
            test_f.write(id)
        test_f.close()

        # src = tmp_dir + img_name
        # dst = 'data/test/' + img_name[:-4] + '.jpg'
        # im = Image.open(src)
        # rgb_im = im.convert('RGB')
        # rgb_im.save(dst)
