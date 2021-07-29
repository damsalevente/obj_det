import torch
import torch.nn as  nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from selective import extract_candidates, extract_iou
from torchvision import transforms, utils, models
from torch import optim

import os
from torch.utils.data import Dataset, DataLoader


device = 'cpu'
## model 
vgg_backbone = models.vgg16(pretrained=True)
vgg_backbone.classifier = nn.Sequential()
for param in vgg_backbone.parameters():
    param.requires_grad = False
    vgg_backbone.eval().to(device)

class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = 25088
        self.backbone = vgg_backbone
        self.cls_score = nn.Linear(feature_dim, \
                14) # this is not the way to do it 
        self.bbox = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 4),
                nn.Tanh(),
                )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()
    
    def forward(self, x):
        feat = self.backbone(x)
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat)
        return cls_score, bbox

    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        ixs, = torch.where(labels != 0)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        self.lmb = 10.0
        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss


def train_batch(inputs, model, optimizer, criterion):
    input, clss, deltas = inputs
    model.train()
    optimizer.zero_grad()
    _clss, _deltas = model(input)
    loss, loc_loss, regr_loss = criterion(_clss, _deltas, clss, deltas)
    accs = clss == decode(_clss)
    loss.backward()

    optimizer.step()
    return loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()

@torch.no_grad()
def validate_batch(inputs, model, criterion):
    input, clss, deltas = inputs
    with torch.no_grad():
        model.eval()
        _clss,_deltas = model(input)
        loss,loc_loss,regr_loss = criterion(_clss, _deltas, 
                clss, deltas)
        _, _clss = _clss.max(-1)
        accs = clss == _clss
        return _clss,_deltas,loss.detach(),loc_loss, regr_loss, accs.cpu().numpy()


## dataset 
idx = ['frame_idx',
        'target_id', # which object is it
        'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
        'score', 'in_view', 'occlusion', 'object_category']


class MyDataSet(Dataset):
    '''
    This class's task is to extract the images and the annotations from the data folder
    Later, this data will be used to build the dronedataset, see below
    '''
    def __init__(self, image_folder, label_text): 
        self.classnames = []
        self.image_folder = image_folder # image fodler, just open the files with os.listdir
        self.label_text = label_text # label folder
        files = sorted([x for x in os.listdir(self.image_folder) if '.jpg' in x])
        self.data = []
        self.images = []
        for f in os.listdir(self.label_text):
            if 'gt_whole.txt' in f and "M0202" in f:
                with open(self.label_text+'/'+f) as filerino:
                    for line in filerino:
                        if line:
                            self.data.append([f,line.replace('\n', '')])

        for (dirpath, dirnames, filenames) in os.walk(self.image_folder):
            for filename in filenames:
                if filename.endswith('.jpg') and "M0202" in dirpath:
                    self.images.append([filename, dirpath, os.sep.join([dirpath, filename])])

        print(len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #i should need to find every object for one element
        img_path = self.images[idx]
        image = cv2.imread(img_path[2], 1)[...,::-1]
        h, w, _ = image.shape
        boxes = []
        classes = []
        for element in self.data:
            # same folder
            if img_path[1].split('/')[2] ==  element[0].split('_')[0]:
                #same frame
                # find every element for that image
                if int(element[1].split(',')[0]) == int(img_path[0].replace('img','').replace('.jpg','')):
                    #open the image and the bounding boxes 
                    _box = np.array([int(x) for x in element[1].split(',')[2:6]])
                    print(_box)
                    boxes.append(_box.astype(np.uint16))
                    classes.append(int(element[-1][-1])) # last element is the class id
                    # there should be an object reference id too but whatever rn
        return image, boxes, classes, img_path[2]
                    
normalize= transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])

def preprocess_img(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.to(device).float()

def decode(_y):
    _, preds = _y.max(-1)
    return preds


class DroneDataset(Dataset):
    def __init__(self,  fpaths, rois, labels, deltas, gtbbs):
        self.fpaths = fpaths
        self.rois = rois
        self.labels  = labels
        self.deltas  =deltas
        self.gtbbs = gtbbs

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread(fpath, 1)[...,::-1]
        H, W, _ = image.shape
        sh = np.array([W,H,W,H])
        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        bbs = (np.array(rois)*sh).astype(np.uint16)
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        crops = [image[y:Y,x:X] for (x,y,X,Y) in bbs]
        return image,crops,bbs,labels,deltas,gtbbs,fpath

    def collate_fn(self, batch):
        inp, rois, rixs, labels, deltas =[],[],[],[],[]
        for ix in range(len(batch)):
            image, crops, image_bbs, image_labels, image_deltas, image_gt_bbs, image_fpath = batch[ix]
            crops = [cv2.resize(crop, (224,224)) for crop in crops]
            inp.extend([preprocess_img(crop/255.)[None] for crop in crops])
            labels.extend([c for c in image_labels])
            deltas.extend(image_deltas)
        inp = torch.cat(inp).to(device)
        labels = torch.Tensor(labels).long().to(device)
        deltas = torch.Tensor(deltas).float().to(device)
        return inp, labels, deltas



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.cnn = nn.Sequential(
              nn.Conv2d(1, 6, 5),
              nn.ReLU(),
              nn.MaxPool2d(2),
              nn.Conv2d(6, 16, 5),
              nn.ReLU(),
              nn.MaxPool2d(2),
              nn.Flatten())
        self.fully_connected = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
              nn.ReLU(),
              nn.Linear(120,84),
              nn.ReLU(),
              nn.Linear(84,10),
              nn.ReLU())

    def forward(self,x):
        return self.fully_connected(self.cnn(x))

def show(self, elem, filename='roi.png'): 
    img, bbs, classes, _ = elem
    copy = img.copy()
    for bb in bbs:
        x,y,w,h = bb[0], bb[1], bb[2], bb[3]
        cv2.rectangle(copy,(x,y),(x+w,y+h),(36,255,12),2)
    cv2.imwrite('ROI_.png',copy)

def prep_data(dataset, N= 500):
    FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [],[],[],[],[],[]
    for idx, (img, bbs, classes, fpath) in enumerate(dataset):
        if idx == N:
            break
        H,W,_ = img.shape
        candids = extract_candidates(img)
        candids = np.array([(x,y,x+w,y+h) for (x,y,w,h) in candids])
        ious, rois, clss, deltas = [], [], [], []
        ious = np.array([[extract_iou(candidate, _bb_) for candidate in candidats] for _bb_ in bbs]).T

        for jx, candidate in enumerate(candids):
            cx, cy, cX, cY = candidate
            candidate_ious = ious[jx]
            best_iou_at = np.argmax(condidate_ious)
            best_iou = candidate_ious[best_iou_at]
            best_bb = _x,_y,_X,_Y = bbs[best_iou_at]
            if best_iou > 0.3:
                clss.append(best_iout_at)
            else:
                clss.append(13) #background

            delta = np.array([_x-cx, _y-cy, _X-cX, _Y-cY]) / np.array([W,H,W,H])
            deltas.append(delta)
            rois.append(candidate / np.array([W,H,W,H]))

            FPATHS.append(fpath)
            IOUS.append(ious)
            ROIS.append(rois)
            CLSS.append(clss)
            DELTAS.append(deltas)
            GTBBS.append(bbs)
        return FPATHS, IOUS, ROIS, CLSS, DELTAS, GTBBS

if __name__ == '__main__':
    dataset = MyDataSet('./UAV-benchmark-M/', './UAV-benchmark-MOTD_v1.0/GT/')
    N = 5 
    n_train = 9*N //10
    for img, bbs, classes, fpath in dataset:
        H, W, _ = img.shape
        newfile = fpath.replace('jpg', 'txt')
        with open(newfile, 'w') as f:
            for (bb, cl) in zip(bbs, classes):
                print(bb[0])
                bb[0] = float(bb[0] + bb[0] + bb[2]) / 2.0 
                bb[1] = float(bb[1] + bb[1] + bb[3]) / 2.0 
                x,y,w,h = bb[0], bb[1], bb[2], bb[3]
                f.write("{} {} {} {} {} \n".format(cl, x/W, y/H, w/W, h/H))
    exit(1)
    FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [],[],[],[],[],[]
    for idx, (img, bbs, classes, fpath) in enumerate(dataset):
        if idx == N:
            break
        H,W,_ = img.shape
        candids = extract_candidates(img)
        candids = np.array([(x,y,x+w,y+h) for (x,y,w,h) in candids])
        ious, rois, clss, deltas = [], [], [], []
        ious = np.array([[extract_iou(candidate, _bb_) for candidate in candids] for _bb_ in bbs]).T

        for jx, candidate in enumerate(candids):
            cx, cy, cX, cY = candidate
            candidate_ious = ious[jx]
            best_iou_at = np.argmax(candidate_ious)
            best_iou = candidate_ious[best_iou_at]
            best_bb = _x,_y,_X,_Y = bbs[best_iou_at]
            if best_iou > 0.3:
                clss.append(best_iou_at)
            else:
                clss.append(13)

            delta = np.array([_x-cx, _y-cy, _X-cX, _Y-cY]) / np.array([W,H,W,H])
            deltas.append(delta)
            rois.append(candidate / np.array([W,H,W,H]))

            FPATHS.append(fpath)
            IOUS.append(ious)
            ROIS.append(rois)
            CLSS.append(clss)
            DELTAS.append(deltas)
            GTBBS.append(bbs)

    train_ds = DroneDataset(FPATHS[:n_train], ROIS[:n_train], CLSS[:n_train], DELTAS[:n_train], GTBBS[:n_train])
    test_ds = DroneDataset(FPATHS[n_train:], ROIS[n_train:], CLSS[n_train:], DELTAS[n_train:], GTBBS[n_train:])
    from torch.utils.data import TensorDataset, DataLoader
    train_loader = DataLoader(train_ds, batch_size=1, 
            collate_fn=train_ds.collate_fn, 
            drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=1, 
            collate_fn=test_ds.collate_fn, 
            drop_last=True)

    rcnn = RCNN()
    criterion = rcnn.calc_loss
    optimizer = optim.SGD(rcnn.parameters(), lr = 1e-3)
    n_epochs = 1
    _n = len(train_loader)
    for ix, inputs in enumerate(train_loader):
        loss, loc_loss,regr_loss,accs = train_batch(inputs, rcnn, optimizer, criterion)
    _n = len(test_loader)
    for ix,inputs in enumerate(test_loader):
        _clss, _deltas, loss, loc_loss, regr_loss, accs = validate_batch(inputs, rcnn, criterion)
