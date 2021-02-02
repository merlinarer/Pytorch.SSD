'''Object Detection using SSD'''

# Importing the libraries
import torch
import cv2
from modeling import build_detector
import glob
from tqdm import tqdm
from utils.checkpoint import load_model
import importlib
import os
from data import BaseTransform
from pycocotools.coco import COCO

labelmap = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

'''Defining the function to perform Detections'''


class COCOMap(object):
    def __init__(self,
                 root='/home/data/COCO/coco/',
                 type='val2017'):
        self.root = root
        self.coco = COCO(os.path.join(root, 'annotations', 'instances_'
                                      + type + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        self.classes, self.coco_labels, self.coco_labels_inverse, self.labels = {}, {}, {}, {}
        for c in categories:
            self.labels[len(self.classes)] = c['name']
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)


def detect(frame, net, transform):
    coco = COCOMap()
    label = coco.labels
    height, width, _ = frame.shape
    frame_t, _, _ = transform(frame)
    frame_t = torch.tensor(frame_t, dtype=torch.float32).permute(2, 0, 1)
    x = frame_t.unsqueeze(0).cuda()
    y = net(x)
    # from IPython import embed
    # embed()
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
                        label[i - 1],
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
config_file = 'config/ssd.coco.yml'
cfgname = os.path.basename(config_file).split('.')[0]
cfg = importlib.import_module('config.{}_defaults'.format(cfgname)).Cfg
cfg.merge_from_file(config_file)
cfg.freeze()
model = build_detector(cfg)
state_dict = load_model('output/cocotrainval35k_bs32_singleGPU/epoch_40.pth')
model.load_state_dict(state_dict)
model.cuda()
model.head.nms = True
reader = glob.glob('/home/data/COCO/coco/val2017/' + '*jpg')

for i, frame in enumerate(tqdm(reader)):
    frame = cv2.imread(frame)
    frame = detect(frame, model.eval(), BaseTransform())
    cv2.imwrite('vis/' + str(i) + '.jpg', frame)
