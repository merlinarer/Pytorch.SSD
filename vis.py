'''
Code Updated By: Sayak Banerjee
'''

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
    x = frame_t.unsqueeze(0).cuda()  # -> Expand the Dimensions to include the batch size
    y = net(x)  # -> Apply Network to model
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]

    for i in range(detections.size(1)):  # -> detections.size(1) = int(num_of_classes)
        j = 0
        while detections[0, i, j, 0] >= 0.4:  # -> confidence_score > 0.6
            pt = (detections[0, i, j, 1:] * scale).numpy()
            # print(pt)
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0),
                          2)  # -> Draw the Rectangle
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                        2, cv2.LINE_AA)  # -> Putting the Label
            j += 1
            if j >= detections.shape[2]:
                break
    return frame


def load_network(network):
    save_path = 'output/baseline/epoch_395.pth'
    checkpoint = torch.load(save_path)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['model'].items()}
    network.load_state_dict(state_dict)
    return network


'''Creating the SSD neural network'''
model = SSD(num_classes=21, phase='test')
model = load_network(model)
model.cuda()

'''Creating the transformation'''
from data import BaseTransform

'''Doing some Object Detection on a video'''
reader = glob.glob('/home/workspace/merlin/data_dirs/VOCdevkit/VOC2007/JPEGImages/' + '*jpg')

for i, frame in enumerate(tqdm(reader)):
    frame = cv2.imread(frame)
    frame = detect(frame, model.eval(), BaseTransform())
    cv2.imwrite('vis/' + str(i) + '.jpg', frame)