import torch

class IoU_assigner(object):
    def __init__(self, pos_iou_thr=0.5, neg_iou_thr=0.5, eps=1e-6):
        self.pos_iou_thr = 0.5
        self.neg_iou_thr = 0.5
        self.eps = eps

    def IoU_cal(self, gt_bboxes, anchors):
        assert gt_bboxes.shape[-1] in [0,4]
        assert anchors.shape[-1] in [0,4]

        row = gt_bboxes.size(-2)
        col = anchors.size(-2)

        if row*col==0:
            return gt_bboxes.new(gt_bboxes.shape[:-2] + (row, col))
        gt_area = (gt_bboxes[...,2] - gt_bboxes[...,0]) * (gt_bboxes[...,3] - gt_bboxes[...,1])
        anchor_area = (anchors[...,2] - anchors[...,0]) * (anchors[...,3] - anchors[...,1])

        lt = torch.max(gt_bboxes[...,:,None,:2], anchors[...,None,:,:2]) # [row, col, 2]
        rb = torch.min(gt_bboxes[...,:,None,2:], anchors[...,None,:,2:])

        cut = (rb - lt).clamp(min=0)
        overlap = cut[...,0] * cut[...,1]

        union = gt_area[..., None] + anchor_area[...,None,:] - overlap
        eps = union.new_tensor([self.eps])
        union = torch.max(union, eps)
        return overlap / union


    def assign(self, gt_bboxes, anchors, gt_labels=None):
        """
        assign anchors with gt_bboxes,
        1. default -1
        2. background 0
        3. assigned >0 (gt_idx+1)
        :param gt_bboxs: Tensor: num_gt, 4
        :param anchors: anchors: num_anchors, 4
        :return:
        """
        IoU = self.IoU_cal(gt_bboxes, anchors) # [num_gt, num_anchor]
        num_gt, num_anchor = IoU.shape[-2:]
        assign_gt_idx = IoU.new_full((num_anchor,), -1, dtype=torch.long)
        if num_anchor==0 or num_gt==0:
            max_over = IoU.new_zeros((num_anchor, ))
            if not num_gt:
                assign_gt_idx[:] = 0
            if gt_labels is None:
                assign_labels = None
            else:
                assign_labels = IoU.new_full((num_anchor, ), -1, dtype=torch.long)
            return assign_gt_idx, assign_labels

        max_over, argmax_over = IoU.max(dim=0) #[num_anchor]
        gt_max_over, gt_argmax_over = IoU.max(dim=1) #[num_gt]

        assign_gt_idx[(max_over>=0) & (max_over<self.neg_iou_thr)] = 0

        pos_idx = max_over>=self.pos_iou_thr
        assign_gt_idx[pos_idx] = argmax_over[pos_idx] + 1 # assigned with idx+1

        for i in range(num_gt):
            if gt_max_over[i]>self.pos_iou_thr:
                assign_gt_idx[gt_argmax_over[i]] = i+1
        if gt_labels is not None:
            assign_labels = assign_gt_idx.new_full((num_anchor, ), -1, dtype=torch.long)
            pos_idx = torch.nonzero(assign_gt_idx>0, as_tuple=False).squeeze() # [num_pos_anchor]
            if pos_idx.numel()>0:
                # 如果匹配的anchor中满足正样本条件的数量大于0，把其中正样本的对应位置
                # 的label赋给assigned_label减一回归原来的gt框的下标, 得到真实类别
                assign_labels[pos_idx] = gt_labels[assign_gt_idx[pos_idx] - 1]
        else:
            assign_labels = None
        return assign_gt_idx, assign_labels