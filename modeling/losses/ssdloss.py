import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from modeling.utils.coder import encode, decode
from modeling.utils.iou import intersect, cal_iou_matrix, point_form


__all__ = [
    "SSDLoss",
]


class SSDLoss:
    def __init__(self,
                 num_classes=21,
                 overlap_thresh=0.5,
                 variances=None,
                 neg_pos_ratio=3):
        if variances is None:
            variances = [0.1, 0.2]
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.variances = variances
        self.neg_pos_ratio = neg_pos_ratio

    def __call__(self, pred_bb, pred_label, gt_bb, gt_label, anchor):
        """SSDLoss
                Args:
                    pred_bb shape: (batch_size,num_anchor,4)
                    pred_label shape: (batch_size,num_anchor,num_classes)
                    gt_bb shape: (batch_size,num_anchor,4)
                    gt_label shape: (batch_size,num_anchor)
                """
        assert len(pred_bb) == len(pred_label) == len(gt_bb) == len(gt_label)
        bs = len(pred_bb)
        num_anchor = anchor.shape[0]

        loc_t = torch.Tensor(bs, num_anchor, 4)
        conf_t = torch.LongTensor(bs, num_anchor)
        for img_idx in range(bs):
            num_gt = gt_bb[img_idx].shape[0]
            iou_matrix = cal_iou_matrix(gt_bb[img_idx].data, point_form(anchor.data))  # shape: (num_gt, num_anchor)

            best_anchor_iou, best_anchor_idx = iou_matrix.max(1)  # shape: (num_gt)
            best_gt_bb_iou, best_gt_bb_idx = iou_matrix.max(0)  # shape: (num_anchor)

            for j in range(num_gt):
                best_gt_bb_idx[best_anchor_idx[j]] = j

            anchor_gt = gt_bb[img_idx][best_gt_bb_idx]  # shape: (num_anchor, 4), assign every anchor to gts
            anchor_label = gt_label[img_idx][best_gt_bb_idx].data + 1  # shape: (num_anchor), 0 for bg
            best_gt_bb_iou = best_gt_bb_iou.index_fill(0, best_anchor_idx, 1)  # ensure every gt has a anchor
            anchor_label[best_gt_bb_iou < 0.5] = 0  #
            # anchor_label[best_anchor_idx] = 1

            # make the label for net output
            loc_t[img_idx] = encode(anchor_gt, anchor, self.variances)
            # xy = ((anchor_gt[:, 2:] + anchor_gt[:, :2]) / 2 - anchor[:, :2]) / (self.variances[0] * anchor[:, 2:])
            # wh = torch.log((anchor_gt[:, 2:] - anchor_gt[:, :2]) / anchor[:, 2:]) / self.variances[1]
            # loc_t[img_idx] = torch.cat([xy, wh], 1)  # [num_anchor, 2 + 2]

            conf_t[img_idx] = anchor_label

        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()

        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0  # (bs, num_anchor)
        loc_t = loc_t[pos]
        from IPython import embed
        embed()
        loc_p = pred_bb[pos]
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # cls loss
        pred_label = pred_label.view(-1, self.num_classes)

        # def log_sum_exp(x):
        #     x_max = x.data.max()
        #     return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max

        # loss_all_bb = torch.log(torch.sum(torch.exp(pred_label), 1, keepdim=True)) \
        #               - pred_label.gather(1, conf_t.view(-1, 1))  # Shape: [bs * num_anchor, 1]
        # loss_all_bb = log_sum_exp(pred_label) - pred_label.gather(1, conf_t.view(-1, 1))
        loss_all_bb = F.cross_entropy(pred_label, conf_t.view(-1), reduction='none')

        loss_all_bb = loss_all_bb.view(bs, -1)  # Shape: [num, num_priors]
        loss_all_bb[pos] = 0
        _, loss_idx = loss_all_bb.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=num_anchor - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        pred_label = pred_label.reshape(bs, num_anchor, -1)

        targets_label = conf_t[pos + neg]

        # pos = pos.unsqueeze(2).expand_as(pred_label)
        # neg = neg.unsqueeze(2).expand_as(pred_label)

        conf_p = pred_label[pos + neg].view(-1, self.num_classes)
        loss_c = F.cross_entropy(conf_p, targets_label, reduction='sum')

        N = num_pos.data.sum()
        if N == 0:
            loss_l = 0
            loss_c = 0
        else:
            loss_l /= N
            loss_c /= N

        # from IPython import embed
        # embed()

        # import math
        # if math.isnan((loss_l + loss_c).cpu()):
        #     from IPython import embed
        #     embed()

        return loss_l + loss_c


if __name__ == '__main__':
    # torch.manual_seed(0)
    loss = SSDLoss()
    pred_bb = torch.randn(16, 8732, 4)
    gt_label = np.random.randint(0, 20, (16, 8732))
    gt_label = torch.tensor(gt_label, dtype=torch.float)
    pred_label = torch.randn(16, 8732, 21)
    gt_bb = torch.randn(16, 10, 4)
    # gt_label = torch.randn(16, 10)
    testloss = loss(pred_bb, pred_label, gt_bb, gt_label)
    print(testloss)
