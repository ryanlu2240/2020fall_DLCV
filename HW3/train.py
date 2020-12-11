import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json
import glob

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from itertools import groupby
import cv2
from pycocotools import mask as m


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train_images"))))
        self.coco = COCO("data/pascal_train.json")
        self.list_keys = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        # load images ad masks
        img_id = self.list_keys[idx]
        img_info = self.coco.loadImgs(ids=img_id)
        file_name = img_info[0]['file_name']
        img_path = os.path.join(self.root, "train_images", file_name)
        img = Image.open(img_path).convert("RGB")

        #get mask
        annids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annids)
        labels = []
        masks = []
        boxes = []
        areas = []
        iscrowds = []
        for i in range(len(annids)):
            mask = self.coco.annToMask(anns[i])
            masks.append(mask)
            cate = anns[i]['category_id']
            labels.append(cate)
            xmin = anns[i]['bbox'][0]
            xmax = xmin + anns[i]['bbox'][3]
            ymin = anns[i]['bbox'][1]
            ymax = ymin + anns[i]['bbox'][2]
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
            area = anns[i]['area']
            areas.append(area)
            iscrowd = anns[i]['iscrowd']
            iscrowds.append(iscrowd)

        # get bounding box coordinates for each mask
        num_objs = len(annids)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.int64)
        iscrowds = torch.as_tensor(iscrowds, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowds

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# use our dataset and defined transformations
dataset = VOCDataset('data', get_transform(train=True))
dataset_test = VOCDataset('data', get_transform(train=False))

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

device = torch.device('cuda:4') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 21

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=10,
                                               gamma=0.1)


def inference(epoch):
    cocoGt = COCO("data/test.json")
    coco_dt = []

    transform = transforms.Compose([
        transforms.ToPILImage(),
    	  transforms.ToTensor(),
    	  ]
    )

    for imgid in cocoGt.imgs:
        image = cv2.imread("data/test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1] # load image
        image = transform(image)
        model.eval()
        with torch.no_grad():
            prediction = model([image.to(device)])

        n_instances = len(prediction)
        if len(prediction[0]['labels']) > 0:
            for i in range(n_instances): # Loop all instances
            # save information of the instance in a dictionary then append on coco_dt list
                pred = {}
                pred['image_id'] = imgid # this imgid must be same as the key of test.json
                pred['category_id'] = int(prediction[0]['labels'][i])
                arr_mask = prediction[0]['masks'][i,:,:,:].cpu().numpy()
                #print(arr_mask.shape)
                arr_mask=np.reshape(arr_mask,[arr_mask.shape[1],arr_mask.shape[2]])
                #print(arr_mask.shape)
                mask = arr_mask.tolist()
                jj=0
                for j in mask:
                    kk=0
                    for k in j:
                        if k>0:
                            mask[jj][kk]=1
                        kk = kk+1
                    jj = jj+1
                mask = np.asarray(mask, dtype=np.uint8)

                b_a = np.asfortranarray(np.array(mask==1, dtype=np.bool))
                mm = m.encode(b_a)['counts'].decode("utf-8")
                pred['segmentation'] = {'counts': str(mm), 'size': list(arr_mask.shape)}
                pred['score'] = float(prediction[0]['scores'][i])
                coco_dt.append(pred)

    file_name = '0616066_test_' + str(epoch) + '.json'

    with open(file_name, "w") as f:
        json.dump(coco_dt, f)

num_epochs = 60

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset

    if (epoch+1)%5 ==0:
        evaluate(model, data_loader_test, device=device)
        inference((epoch+1))
