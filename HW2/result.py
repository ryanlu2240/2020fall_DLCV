import json
import cv2
from PIL import Image


def find_idx(id):
    with open('result_3.json') as f:
        data = json.load(f)
    for i in range(len(data)):
        if(int(data[i]['filename'][9:-4]) == id):
            return i

with open('result_3.json') as f:
    data = json.load(f)


result = []
for i in range(1,len(data)+1):
    idx = find_idx(i)
    print(i,idx)
    file_name = data[idx]['filename']
    im = cv2.imread(file_name)
    height = im.shape[0]
    width = im.shape[1]
    bbox = []
    score = []
    label = []
    for item in  data[idx]['objects']:
        y1 = item['relative_coordinates']['center_y']*height - (item['relative_coordinates']['height']*height) /2
        x1 = item['relative_coordinates']['center_x']*width - (item['relative_coordinates']['width']*width)/2
        y2 = item['relative_coordinates']['center_y']*height + (item['relative_coordinates']['height']*height)/2
        x2 = item['relative_coordinates']['center_x']*width + (item['relative_coordinates']['width']*width)/2
        y1 = int(round(y1))
        x1 = int(round(x1))
        y2 = int(round(y2))
        x2 = int(round(x2))
        bbox.append((y1, x1, y2, x2))
        label.append(int(item['name']))
        score.append(item['confidence'])
    result_dict = {}
    result_dict['bbox'] = bbox
    result_dict['score'] = score
    result_dict['label'] = label
    result.append(result_dict)

with open('0616066.json', 'w') as outfile:
    json.dump(result, outfile)

with open('0616066.json') as f:
    data = json.load(f)
print(len(data))
for i in range(2):
    print(data[i])
