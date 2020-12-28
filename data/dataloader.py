#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import logging
from PIL import Image
import numpy as np
import glob
import os
from tabulate import tabulate
import json
import numpy as np
import xml.etree.ElementTree as ET
import cv2


# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, root=None, transform=None, type='train'):
        self.type = type
        with open(root, 'r') as f:
            datalist = json.load(f)

        if self.type == 'train':
            self.datapath = datalist[:int(0.9 * len(datalist))]
        elif self.type == 'val':
            self.datapath = datalist[int(0.9 * len(datalist)):]
        elif self.type == 'test':
            self.datapath = datalist
        else:
            RuntimeError

        self.len = len(self.datapath)

        self.transform = transform

        table = [["{}".format(self.type), self.len]]
        headers = ['stage', 'len']
        datainfo = tabulate(table, headers, tablefmt="grid")
        logger = logging.getLogger('merlin.baseline.dataset')
        logger.info('\n' + datainfo)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.datapath[index]['img']
        label = self.datapath[index]['label']

        try:
            img = Image.open(img_path)
        except:
            print(img_path)
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        return (img, int(label))


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        # img.show()
        # resize
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


# VOC Dataset
class MyVOCDataset(Dataset):
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, root=None, transform=None, type='train'):
        self.type = type
        self.root = root
        self.list_file = []
        self.label = []
        self.transform = transform
        image_sets_file = os.path.join(self.root, "ImageSets", "Main", "trainval.txt")
        with open(image_sets_file, 'r') as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                path = os.path.join(self.root, 'JPEGImages', line + '.jpg')
                assert os.path.isfile(path), path
                self.list_file.append(path)
        # self.list_file = self.list_file[:100]
        self.class_dict = {k: v for v, k in enumerate(self.CLASSES)}

        self.len = len(self.list_file)

        table = [["{}".format(self.type), self.len]]
        headers = ['stage', 'len']
        datainfo = tabulate(table, headers, tablefmt="grid")
        logger = logging.getLogger('merlin.baseline.dataset')
        logger.info('\n' + datainfo)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index <= self.len, 'index range error'
        img_path = self.list_file[index]

        try:
            img = cv2.imread(img_path)
            h, w, c = img.shape
        except:
            print(img_path)
            print('Corrupted image for %d' % index)
            return self[index + 1]

        box, label = self.labelpaser(os.path.basename(img_path).rstrip('.jpg'), h, w)

        if self.transform is not None:
            img, box, label = self.transform(img, box, label)

        assert len(box) == len(label)

        return torch.tensor(img).permute(2, 0, 1), box, label

    def labelpaser(self, img, h, w):
        annotation_file = os.path.join(self.root, "Annotations", "%s.xml" % img)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1/w, y1/h, x2/w, y2/h])
            labels.append(self.class_dict[class_name])

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))
