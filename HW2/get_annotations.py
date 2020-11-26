import numpy as np
import cv2
import os
import pandas as pd
import h5py

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

f = h5py.File('digitStruct.mat','r') 
for j in range(f['/digitStruct/bbox'].shape[0]):
    print(j)
    img_name = get_name(j, f)
    row_dict = get_bbox(j, f)
    print(img_name, row_dict)
    # row_dict['img_name'] = img_name
    # all_rows.append(row_dict)
    # bbox_df = pd.concat([bbox_df,pd.DataFrame.from_dict(row_dict,orient = 'columns')])
