import os
import numpy as np
import torch
from itertools import product as product
import math


from math import sqrt as sqrt
from itertools import product as product
import torch

voc = {
    'image_size': 300,
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'self.voc',
}


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self):
        super(PriorBox, self).__init__()
        self.image_size = voc['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(voc['aspect_ratios'])
        self.variance = voc['variance'] or [0.1]
        self.feature_maps = voc['feature_maps']
        self.min_sizes = voc['min_sizes']
        self.max_sizes = voc['max_sizes']
        self.steps = voc['steps']
        self.aspect_ratios = voc['aspect_ratios']
        self.clip = True
        self.version = voc['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class AnchorGenerator:
    voc = {
        'image_size': 300,
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'self.voc',
    }

    def __init__(self):
        pass

    def delta(self, size):
        row = np.linspace(0, size - 1, size)
        y, x = np.meshgrid(row, row)
        return x, y

    def __call__(self):
        total_anchor = []
        for k, fs in enumerate(self.voc['feature_maps']):
            temp = []
            y_offset, x_offset = self.delta(fs)
            f_k = self.voc['image_size'] / self.voc['steps'][k]
            # unit center x,y
            cx = 0.5 / f_k
            cy = 0.5 / f_k
            # aspect_ratio: 1
            # rel size: min_size
            s_k = self.voc['min_sizes'][k] / self.voc['image_size']
            temp += [cx, cy, s_k, s_k]
            # aspect_ratio: 1
            # rel size: math.sqrt(s_k * s_(k+1))
            s_k_prime = math.sqrt(s_k * (self.voc['max_sizes'][k] / self.voc['image_size']))
            temp += [cx, cy, s_k_prime, s_k_prime]
            # rest of aspect ratios
            for ar in self.voc['aspect_ratios'][k]:
                temp += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                temp += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
            temp = np.reshape(np.array(temp), (-1, 4))
            anchor = np.concatenate([x_offset[..., np.newaxis],
                                     y_offset[..., np.newaxis],
                                     np.zeros_like(x_offset)[..., np.newaxis],
                                     np.zeros_like(x_offset)[..., np.newaxis]], 2)

            num = 4 if k in [0, 4, 5] else 6
            anchor = np.expand_dims(anchor, 2).repeat(num, axis=2)  / f_k
            anchor += temp[np.newaxis, np.newaxis, ...]

            total_anchor.append(anchor.reshape(-1, 4))

        # from IPython import embed
        # embed()
        return np.clip(np.concatenate(total_anchor, 0), a_min=0, a_max=1)


if __name__ == '__main__':
    import cv2
    num = 100
    a = AnchorGenerator()
    print(a().shape)  # (8732, 4)
    print(a()[0])  # (8732, 4)
    pt = a() * 300
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(num):
        cv2.rectangle(frame, (int(pt[i][0]), int(pt[i][1])), (int(pt[i][2]), int(pt[i][3])), (255, 255, 255), 1)  # -> Draw the Rectangle

    a = PriorBox()
    an = a.forward()
    print(an.shape)
    pt = an * 300
    frame1 = np.zeros((300, 300, 3), dtype=np.uint8)
    import cv2
    for i in range(num):
        cv2.rectangle(frame1, (int(pt[i][0]), int(pt[i][1])), (int(pt[i][2]), int(pt[i][3])), (255, 255, 255),
                      1)  # -> Draw the Rectangle
    frame = np.hstack([frame, frame1])
    cv2.imshow('im', frame)
    cv2.waitKey()
