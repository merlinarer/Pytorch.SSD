import torch
from .coder import decode
from torchvision.ops import nms


def ori_nms(bboxes, scores, threshold=0.5, topk=200):
    """NMS Algorithm
        Args:
            boxes: Shape: [num_priors,4].
            scores: Shape:[num_priors].
            overlap: The overlap thresh.
        Return:
            The indices of the kept boxes w.r.t num_priors.
        """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)  # [N,]
    _, order = scores.sort(0, descending=True)  # descending
    order = order[:topk]

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i].data)  # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i].data)
        xx2 = x2[order[1:]].clamp(max=x2[i].data)
        yy2 = y2[order[1:]].clamp(max=y2[i].data)
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]

        iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze()  # idx [N-1,], order [N,]
        if idx.numel() == 0:
            break
        order = order[idx + 1]
    return torch.LongTensor(keep), len(keep)


class Detect:
    def __init__(self,
                 num_classes=21,
                 conf_thresh=0.01,
                 nms_thresh=0.5,
                 variance=None,
                 prior=None,
                 topk=200):
        if variance is None:
            variance = [1, 1]
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.topk = topk
        self.prior = prior

    def __call__(self, loc_data, conf_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4]
        Return:
             Shape: [batch,num_classes,5] 0 is score, 1234 is bb cords
        """
        bs = loc_data.size(0)  # batch size
        num_priors = self.prior.size(0)
        output = torch.zeros(bs, self.num_classes, self.topk, 5)
        conf_preds = conf_data.view(bs, num_priors,
                                    self.num_classes).transpose(2, 1)
        for i in range(bs):
            decoded_boxes = decode(loc_data[i], self.prior.cuda(), self.variance)
            conf_scores = conf_preds[i].clone()
            # For each class, perform nms, skip bkg=0
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                # apply ori_nms
                # ids, count = ori_nms(boxes, scores, self.nms_thresh, self.topk)
                # output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                #                                    boxes[ids[:count]]), 1)
                # apply torchvision build free cuda nms
                ids = nms(boxes, scores, self.nms_thresh)
                count = [ids.shape[0], self.topk][ids.shape[0] > self.topk]
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)
        return output
