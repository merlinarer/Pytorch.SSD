import os
import numpy as np
import torch
from itertools import product as product
import math


from math import sqrt as sqrt
from itertools import product as product
import torch

def pairs(x):
    assert type(x) != str
    return [x,x]

def xywh_xyxy(anchor):
    x1 = anchor[...,0] - 0.5 * anchor[...,2]
    y1 = anchor[...,1] - 0.5 * anchor[...,3]
    x2 = anchor[...,0] + 0.5 * anchor[...,2]
    y2 = anchor[...,1] + 0.5 * anchor[...,3]
    if type(anchor)==np.ndarray:
        return np.stack([x1, y1, x2, y2], axis=-1)
    elif type(anchor)==torch.Tensor:
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        raise TypeError("anchor should be ndarray type or Tensor type.")

def xyxy_xywh(anchor):
    w = anchor[...,2] - anchor[...,0]
    h = anchor[...,3] - anchor[...,1]
    center_x = anchor[...,0] + 0.5*w
    center_y = anchor[...,1] + 0.5*h
    if type(anchor) == np.ndarray:
        return np.stack([center_x, center_y, w, h], axis=-1)
    elif type(anchor) == torch.Tensor:
        return torch.stack([center_x, center_y, w, h], dim=-1)
    else:
        raise TypeError("anchor should be ndarray type or Tensor type.")

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

class AnchorGenerator(object):
    def __init__(self, strides, ratios, scales=None, base_length=None):
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        self.base_length = [scale for scale in self.scales] if base_length is None else base_length
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_length):

            multi_level_base_anchors.append(
                self.gen_single_feat_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    stride=self.strides))
        return multi_level_base_anchors

    def gen_single_feat_anchors(self,
                                base_len,ratios,scales,stride,):
        w, h = base_len
        w_center = int(stride / 2)
        h_center = int(stride / 2)

        h_ratio = torch.sqrt(ratios)
        w_ratio = 1 / h_ratio

        d_w = w * scales[:, None] * w_ratio[None, :]
        d_h = h * scales[:, None] * h_ratio[None, :]

        d_w = d_w.view(-1)
        d_h = d_h.view(-1)

        anchors = [w_center - 0.5 * d_w,
                   h_center - 0.5 * d_h,
                   w_center + 0.5 * d_w,
                   h_center + 0.5 * d_h,
                   ]

        anchors = torch.stack(anchors, dim=-1)

        return anchors

    def grid_anchors(self, image_size, clamp=True, device='cuda'):
        """
        :argument:
            self.base_anchors: List[Tensor]
            self.anchor_strides: List[int]
        :return:
            multi_grid_anchors: list[Tensor: num_levels, featmap_size, 4]
        """
        image_size_w, image_size_h = image_size if type(image_size) not in [int, float] else pairs(image_size)
        multi_grid_anchors = []
        for num_levels in range(len(self.base_anchors)):
            x = torch.from_numpy(np.array(
                range(0, image_size_w, self.strides[num_levels]))).to(device)
            y = torch.from_numpy(np.array(
                range(0, image_size_h, self.strides[num_levels]))).to(device)
            x_, y_ = self.shift(x,y)
            shift = [x_, y_, x_, y_]
            shift = torch.stack(shift, dim=1)
            shift_=shift.type_as(self.base_anchors[num_levels].to(device))
            shift_anchors = self.base_anchors[num_levels].to(device)[None,:,:]+shift_[:,None,:]
            shift_anchors = shift_anchors.view(-1,4)
            if clamp:
                shift_anchors = shift_anchors.clamp(min=0,max=image_size)
            multi_grid_anchors.append(shift_anchors)
        return multi_grid_anchors

    def shift(self,x,y):
        """
        :param x: Tensor, w of featmap
        :param y: int, h of featmap
        :return: shift: Tensor, wh * 4
        """
        x_ = x.repeat(len(y))
        y_ = y.view(-1,1).repeat(1,len(x)).view(-1)

        return x_, y_

    def single_img_anchors(self, image_size):
        grid_anchors = self.grid_anchors(image_size)
        concat_anchors = torch.cat(grid_anchors,dim=0)
        return concat_anchors


class SSDAnchorGenerator(AnchorGenerator):
    def __init__(self,
                 image_size=300,
                 strides=(8,16,32,64,100,300),
                 ratios=([2],[2,3],[2,3],[2,3],[2,],[2]),
                 size_ratio_range=(0.2, 0.9),
                 clip = False,
                 ):

        self.image_size = image_size
        self.strides = strides
        self.min_size_ratio, self.max_size_ratio = size_ratio_range
        self.clip = clip

        assert self.min_size_ratio in [0.15, 0.2]
        self.min_sizes = []
        self.max_sizes = []
        if self.min_size_ratio == 0.2:
            self.min_sizes.append(0.1 * self.image_size)
        elif self.min_size_ratio == 0.15:
            self.min_sizes.append(0.07 * self.image_size)
        self.max_sizes.append(self.min_size_ratio*self.image_size)
        size_step = int(np.floor((self.max_size_ratio - self.min_size_ratio) * 100) / (len(self.strides) - 2)) #int
        for ratio in range(int(self.min_size_ratio*100), int(self.max_size_ratio*100)+1, size_step):
            self.min_sizes.append(int(self.image_size * ratio / 100))
            self.max_sizes.append(int(self.image_size * (ratio+size_step) / 100))

        anchor_ratios = []
        anchor_scales = []
        for k in range(len(self.strides)):
            scales = [1., np.sqrt(self.max_sizes[k] / self.min_sizes[k])]
            anchor_ratio = [1.]
            for r in ratios[k]:
                anchor_ratio += [1/r, r]  # 4 or 6 ratio
            anchor_ratios.append(torch.Tensor(anchor_ratio))
            anchor_scales.append(torch.Tensor(scales))

        self.scales = anchor_scales
        self.ratios = anchor_ratios
        self.base_length = self.min_sizes
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return [base_anchor.size(0) for base_anchor in self.base_anchors]

    def gen_base_anchors(self,device='cuda'):
        mulit_feat = []
        for i in range(len(self.base_length)):
            base_anchors=self.gen_single_feat_anchors(pairs(self.base_length[i]),
                                                          stride=self.strides[i],
                                                          ratios=self.ratios[i],
                                                          scales=self.scales[i])
            indices = list(range(len(self.ratios[i])))
            indices.insert(1,len(indices))
            base_anchors = torch.index_select(base_anchors.to(device), 0, torch.LongTensor(indices).to(device))
            mulit_feat.append(base_anchors)
        return mulit_feat



class AnchorGenerator_(object):
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

        return np.clip(np.concatenate(total_anchor, 0), a_min=0, a_max=1)


if __name__ == '__main__':
    import cv2
    num = 10
    a = AnchorGenerator_()
    #print(a()[0]*300)  # (8732, 4)
    pt = a() * 300
    qt = xywh_xyxy(pt)
    def f(i):
        return i
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    for t in range(num):
        i = f(t)
        #pt[i][0], pt[i][1], pt[i][2], pt[i][3] = xywh_xyxy(pt[i])
        print(qt[i])
        print(pt[i])
        cv2.rectangle(frame, (int(pt[i][0]), int(pt[i][1])), (int(pt[i][2]), int(pt[i][3])), (255, 255, 255), 1)  # -> Draw the Rectangle
    print('--'*20)
    a = SSDAnchorGenerator()
    grid_anchors = a.single_img_anchors(image_size=(300,300))
    g = grid_anchors
    gt = xyxy_xywh(g).numpy()
    #print(g[0])
    frame1 = np.zeros((300, 300, 3), dtype=np.uint8)
    for t in range(num):
        i = f(t)
        print(g[i])
        print(gt[i])
        cv2.rectangle(frame1, (int(g[i][0]), int(g[i][1])), (int(g[i][2]), int(g[i][3])), (255, 255, 255), 1)
    frame = cv2.hconcat([frame, frame1])
    cv2.imwrite('im.jpg', frame)
    from IPython import embed;embed()
    #cv2.waitKey()
