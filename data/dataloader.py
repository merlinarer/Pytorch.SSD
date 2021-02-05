#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""
import random
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os
from tabulate import tabulate
import json
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from pycocotools.coco import COCO

# Repeat Dataset

class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    # def get_cat_ids(self, idx):
    #     """Get category ids of repeat dataset by index.
    #
    #     Args:
    #         idx (int): Index of data.
    #
    #     Returns:
    #         list[int]: All categories in the image of specified index.
    #     """
    #
    #     return self.dataset.get_cat_ids(idx % self._ori_len)

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len

# Custom Dataset
class CustomDataset(Dataset):
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
        logger = logging.getLogger('evig.detection')
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

        return img, int(label)


# VOC Dataset
class VOCDataset(Dataset):
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
        image_sets_file = os.path.join(self.root, "ImageSets", "Main", "{}.txt".format(type))
        with open(image_sets_file, 'r') as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                path = os.path.join(self.root, 'JPEGImages', line + '.jpg')
                assert os.path.isfile(path), path
                self.list_file.append(path)
        # if type == 'test':
        #     self.list_file = self.list_file[:10]
        self.class_dict = {k: v for v, k in enumerate(self.CLASSES)}

        self.len = len(self.list_file)

        table = [["{}".format(self.type), self.len]]
        headers = ['stage', 'len']
        datainfo = tabulate(table, headers, tablefmt="grid")
        logger = logging.getLogger('evig.detection')
        logger.info('\n' + datainfo)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index <= self.len, 'index range error'
        img_path = self.list_file[index]

        try:
            img = cv2.imread(img_path)
            # img = img[:, :, (2, 1, 0)]  # to RGB
            h, w, c = img.shape
        except:
            print(img_path)
            print('Corrupted image for %d' % index)
            return self[index + 1]
        box, label = self.labelpaser(os.path.basename(img_path).rstrip('.jpg'), h, w)
        if self.transform is not None:
            img, box, label = self.transform(img, box, label)
        assert len(box) == len(label)
        imginfo = {'name': os.path.basename(img_path),
                   'height': h, 'width': w}

        return torch.from_numpy(img).permute(2, 0, 1), box, label, imginfo

    def labelpaser(self, img, h, w):
        annotation_file = os.path.join(self.root, "Annotations", "%s.xml" % img)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        for obj in objects:
            difficult = int(obj.find('difficult').text) == 1
            if difficult:
                continue
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            labels.append(self.class_dict[class_name])

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))


class COCODataset(Dataset):
    def __init__(self, root, transform=None, type='train2017'):
        self.root, self.type = root, type
        self.transform = transform
        self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_'
                                      + self.type + '.json'))
        #self.image_ids = self.coco.getImgIds()[:2000]  # for test
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

        table = [["{}".format(self.type), self.__len__()]]
        headers = ['stage', 'len']
        datainfo = tabulate(table, headers, tablefmt="grid")
        logger = logging.getLogger('evig.detection')
        logger.info('\n' + datainfo)

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        # coco ids is not from 1, and not continue
        # make a new index from 0 to 79, continuely
        # classes:             {names:      new_index}
        # coco_labels:         {new_index:  coco_index}
        # coco_labels_inverse: {coco_index: new_index}
        self.CLASSES, self.coco_labels, self.coco_labels_inverse = {}, {}, {}
        for c in categories:
            self.coco_labels[len(self.CLASSES)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.CLASSES)
            self.CLASSES[c['name']] = len(self.CLASSES)
        self.labels = {}
        for k, v in self.CLASSES.items():
            self.labels[v] = k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_path = self.load_image(index)
        try:
            img = cv2.imread(img_path)
            h, w, c = img.shape
        except:
            print(img_path)
            print('Corrupted image for %d' % index)
            return self[index + 1]
        imginfo = {'name': self.image_ids[index],
                   'height': h, 'width': w}
        box, label = self.load_anns(index, h, w)
        if box.shape[0] == 0:
            if (index + 1) < self.__len__():
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(index - 1)
        if self.transform is not None:
            img, box, label = self.transform(img, box, label)
        assert len(box) == len(label)
        return torch.from_numpy(img).permute(2, 0, 1), box, label, imginfo

    def load_image(self, index):
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        imgpath = os.path.join(self.root,'images', self.type,
                               image_info['file_name'])

        return imgpath

    def load_anns(self, index, h, w):
        annotation_ids = self.coco.getAnnIds(self.image_ids[index], iscrowd=False)
        # anns is num_anns x 5, (x1, x2, y1, y2, new_idx)
        anns = np.zeros((0, 5))

        # skip the image without annoations
        if len(annotation_ids) == 0:
            return anns[:, :-1], anns[:, -1]
        # if len(annotation_ids) == 0:
        #     annotation_ids = self.coco.getAnnIds(self.image_ids[0], iscrowd=False)

        coco_anns = self.coco.loadAnns(annotation_ids)
        for a in coco_anns:
            # skip the annotations with width or height < 1
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            ann = np.zeros((1, 5))
            ann[0, :4] = a['bbox']
            ann[0, 4] = self.coco_labels_inverse[a['category_id']]
            # if self.coco_labels_inverse[a['category_id']] == 0:
            #     print(a['category_id'])
            #     print('coco_labels',self.coco_labels_inverse[a['category_id']])
            anns = np.append(anns, ann, axis=0)

        # (x1, y1, width, height) --> (x1, y1, x2, y2)
        anns[:, 2] += anns[:, 0]
        anns[:, 3] += anns[:, 1]

        anns[:, 0] /= w
        anns[:, 1] /= h
        anns[:, 2] /= w
        anns[:, 3] /= h
        return np.array(anns[:, :4], dtype=np.float32), \
               np.array(anns[:, 4], dtype=np.int64)

    def image_aspect_ratio(self, index):
        image = self.coco.loadImgs(self.image_ids[index])[0]
        return float(image['width']) / float(image['height'])
