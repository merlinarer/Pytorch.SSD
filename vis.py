'''Object Detection using SSD'''

# Importing the libraries
import torch
import cv2
from data.dataloader import MyVOCDataset
from modeling.build import SSD
import imageio
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
from utils.checkpoint import load_model

from data import BaseTransform

labelmap = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

'''Defining the function to perform Detections'''


def detect(frame, net, transform):
    height, width, _ = frame.shape
    frame_t, _, _ = transform(frame)
    # from IPython import embed
    # embed()
    frame_t = torch.tensor(frame_t, dtype=torch.float32).permute(2, 0, 1)
    x = frame_t.unsqueeze(0).cuda()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):  # -> detections.size(1) = int(num_of_classes)
        j = 0
        while detections[0, i, j, 0] >= 0.4:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          (255, 0, 0),
                          2)
            cv2.putText(frame,
                        labelmap[i - 1],
                        (int(pt[0]), int(pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)
            j += 1
            if j >= detections.shape[2]:
                break
    return frame


'''Creating the SSD neural network'''
model = SSD(num_classes=21, nms=True)
model = load_model(model, 'output/baseline/epoch_395.pth')
model.cuda()
reader = glob.glob('/home/workspace/merlin/data_dirs/VOCdevkit/VOC2007/JPEGImages/' + '*jpg')

for i, frame in enumerate(tqdm(reader)):
    frame = cv2.imread(frame)
    frame = detect(frame, model.eval(), BaseTransform())
    cv2.imwrite('vis/' + str(i) + '.jpg', frame)
