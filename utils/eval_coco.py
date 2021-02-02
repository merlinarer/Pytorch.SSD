from __future__ import print_function
import torch
from torch.autograd import Variable

import os
import time
import numpy as np
import json
import gc
import logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def xyxy_xywh(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    return [box[0], box[1], w, h]


class logCOCOeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)

    def summarize(self):
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            logger = logging.getLogger(name="evig.detection")
            logger.info(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class COCOMap(object):
    def __init__(self,
                 root,
                 num_images,
                 type='val2017'):
        self.root = root
        self.coco = COCO(os.path.join(root, 'annotations', 'instances_'
                                      + type + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()
        self.num_images = num_images
        self.type = type
        self.logger = logging.getLogger(name="evig.detection")

    def coco_eval(self):
        annType = ['segm', 'bbox', 'keypoints']
        annType = annType[1]

        # initialize COCO ground truth api
        annFile = os.path.join(self.root, 'annotations', 'instances_'
                               + self.type + '.json')
        cocoGt = COCO(annFile)

        # initialize COCO detections api
        cocoDt = cocoGt.loadRes(self.resFile)
        imgIds = sorted(cocoGt.getImgIds())

        # running evaluation
        ACOCOeval = logCOCOeval(cocoGt, cocoDt, annType)
        ACOCOeval.params.imgIds = imgIds
        ACOCOeval.evaluate()
        ACOCOeval.accumulate()
        ACOCOeval.summarize()
        del self.resFile
        gc.collect()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        # coco ids is not from 1, and not continue
        # make a new index from 0 to 79, continuely
        # classes:             {names:      new_index}
        # coco_labels:         {new_index:  coco_index}
        # coco_labels_inverse: {coco_index: new_index}
        self.classes, self.coco_labels, self.coco_labels_inverse = {}, {}, {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)
        self.labels = {}
        for k, v in self.classes.items():
            self.labels[v] = k

    def __call__(self, net, dataloader):
        self.test_net(net, dataloader)

    def test_net(self, net, dataloader):
        # timers
        _t = {'eval': Timer()}
        # output_dir = self.get_output_dir('temp', self.set_type)
        # det_file = os.path.join(output_dir, 'detections.pkl')
        _t['eval'].tic()
        i = 0
        img_list = []
        results = []
        for img, _, _, imginfo in dataloader:
            h = [info['height'] for info in imginfo]
            w = [info['width'] for info in imginfo]
            imgid = [info['name'] for info in imginfo]

            img = Variable(img.cuda(), requires_grad=False)
            out = net(img).data
            img_list += imgid
            for k in range(out.shape[0]):
                detections = out[k]
                # skip j = 0, because it's the background class
                for j in range(0, detections.size(0)-1):
                #for j in range(1, detections.size(0)):

                    dets = detections[j, :]
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.size(0) == 0:
                        continue
                    boxes = dets[:, 1:]
                    boxes[:, 0] *= w[k]
                    boxes[:, 2] *= w[k]
                    boxes[:, 1] *= h[k]
                    boxes[:, 3] *= h[k]
                    boxes = boxes.cpu().numpy()
                    scores = dets[:, 0].cpu().numpy()
                    for b in range(boxes.shape[0]):
                        result = {
                            "image_id": imgid[k],
                            "category_id": self.coco_labels[j],
                            "bbox": xyxy_xywh(boxes[b].tolist()),
                            "score": scores[b].item()
                        }
                        results.append(result)
                    # from IPython import embed
                    # embed()
                i += 1
                detect_time = _t['eval'].toc(average=False)
                if i%100==0:
                    print('eval: {:d}/{:d} {:.3f}s'.format(i,
                                                       self.num_images, detect_time))

        self.resFile = results
        print('Evaluating detections')
        self.coco_eval()


if __name__ == '__main__':
    pass
