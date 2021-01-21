import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modeling.utils.nms import Detect
from .build import HEAD_REGISTRY
from modeling.utils import anchor
from modeling.utils.coder import encode, decode
from modeling.utils.iou import intersect, cal_iou_matrix, point_form
from modeling.utils.multi_apply import multi_apply, image_to_level
from utils.IoU_assign import IoU_assigner
from modeling.losses.smooth_l1 import smooth_l1
from modeling.utils.anchor import xywh_xyxy

__all__ = [
    "SSDHead",
]


def conv3x3(in_planes, out_planes, padding=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=padding)


@HEAD_REGISTRY.register()
class SSDHead(nn.Module):
    def __init__(self,
                 cfg,
                 nms=False,
                 anchor_num=(4, 6, 6, 6, 4, 4),
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 bbox_cfg={'means':(0., 0., 0., 0.), 'stds':(0.1, 0.1, 0.2, 0.2)}):
        super().__init__()
        assert len(anchor_num) == 6
        self.nms = nms
        self.num_classes = cfg.MODEL.HEADS.NUM_CLASSES #21
        self.variance = cfg.VARIRANCE
        self.in_channels = in_channels
        self.anchor_generator = anchor.SSDAnchorGenerator()
        self.anchor = self.anchor_generator.single_img_anchors(image_size=300)
        reg_convs = []
        cls_convs = []
        for i, ou in enumerate(anchor_num):
            inplanes = self.in_channels[i]
            reg_convs.append(conv3x3(inplanes, ou * 4, padding=1))
            cls_convs.append(conv3x3(inplanes, ou * self.num_classes, padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes=self.num_classes,
                             topk=200,
                             conf_thresh=0.01,
                             nms_thresh=0.45,
                             prior=self.anchor,
                             variance=self.variance)
        self.init_weights()

        self.neg_pos_ratio = 3


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        cls_scores = []
        bbox_preds = []
        self.anchor = self.anchor.cuda()
        for feat, reg_conv, cls_conv in zip(x, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat).permute(0, 2, 3, 1).contiguous())
            bbox_preds.append(reg_conv(feat).permute(0, 2, 3, 1).contiguous())

        if self.nms:
            output = self.detect(torch.cat([o.view(o.size(0), -1, 4) for o in bbox_preds], 1),
                                 self.softmax(torch.cat([o.view(o.size(0), -1, self.num_classes)
                                                         for o in cls_scores], 1))
                                 )
        else:
            output = (
                torch.cat([o.view(o.size(0), -1, 4) for o in bbox_preds], 1),
                torch.cat([o.view(o.size(0), -1, self.num_classes) for o in cls_scores], 1)
            )

        # from IPython import embed
        # embed()

        return output

    def match_single_img(self, flat_anchors, gt_bboxes, gt_labels, batch_size=4):
        """
        match anchors with gt_bboxes in one img
        :param flat_anchors: Tensor: num_anchors, 4
        :param gt_bboxes: Tensor: num_gt, 4
        :param gt_labels: Tensor: num_gt,
        :param batch_size: int
        :return: labels: Tensor: num_anchors, ; num_classes(background) when not assigned with gt
                bbox_targets: Tensor: num_anchors, 4; 0 when not assigned
                pos_idx: list:
        """
        #inside_flags = valid_flags
        anchors = flat_anchors
        assigner = IoU_assigner()
        # [num_anchors, ],
        assign_gt_idx, assign_label = assigner.assign(gt_bboxes, anchors, gt_labels)
        # len(pos_idx)+len(neg_idx)=num_anchors
        pos_idx = torch.nonzero(assign_gt_idx>0, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(assign_gt_idx==0, as_tuple=False).squeeze(-1)
        pos_bboxs = anchors[pos_idx]
        neg_bboxs = anchors[neg_idx]
        pos_gt_bboxes = gt_bboxes[assign_gt_idx[pos_idx]-1]

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        #只有正样本的bbox需要算loss
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full(
            (num_valid_anchors, ), self.num_classes - 1 , dtype=torch.long) #(num_anchors, )
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        if len(pos_idx) > 0:
            pos_bbox_targets = encode(pos_gt_bboxes, pos_bboxs, self.variance)
            bbox_targets[pos_idx, :] = pos_bbox_targets
            bbox_weights[pos_idx, :] = 1.0
            if gt_labels is not None:
                labels[pos_idx] = gt_labels[assign_gt_idx[pos_idx]-1]
            else:
                labels[pos_idx] = 0
            label_weights[pos_idx] = 1.0
        if len(neg_idx) > 0:
            label_weights[neg_idx] = 1.0
        #from IPython import embed;embed()
        return (labels, label_weights, bbox_targets, bbox_weights, pos_idx, neg_idx)

    def match_(self, anchors_list, gt_bboxes, gt_labels, batch_size):
        """
        match anchors with gt_bboxes in a batch of imgs
        :param anchors_list: list[list[Tensor: num_levels, featmap_size, 4]]
        :param gt_bboxes: list[Tensor]
        :param gt_labels: list[Tensor]
        :return:
        """
        num_anchors_level = [anchors.size(0) for anchors in anchors_list[0]]
        concat_anchor_list = []
        #concat_valid_flag_list = []
        for i in range(batch_size):
            concat_anchor_list.append(torch.cat(anchors_list[i])) # list[Tensor: num_total_anchors, 4]
            #concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        #from IPython import embed;embed()
        result = multi_apply(self.match_single_img,
                             concat_anchor_list,
                             gt_bboxes, gt_labels, batch_size=batch_size)
        (all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_pos_idx, all_neg_idx) = result
        num_pos_total = sum([max(inds.numel(),1) for inds in all_pos_idx])
        num_neg_total = sum([max(inds.numel(),1) for inds in all_neg_idx])

        labels_list = image_to_level(all_labels, num_anchors_level)
        label_weights_list = image_to_level(all_label_weights, num_anchors_level)
        bboxes_target_list = image_to_level(all_bbox_targets, num_anchors_level)
        bboxes_weights_list = image_to_level(all_bbox_weights,
                                             num_anchors_level)
        return (labels_list, label_weights_list, bboxes_target_list, bboxes_weights_list, num_pos_total, num_neg_total)

    def loss_single_img(self, cla_scores,  bbox_preds, bbox_weights, anchors, labels, label_weights,
                                           bbox_targets, num_total_samples):
        """

        :param cla_scores: Tensor: num_total_anchor, num_classes+1
        :param bbox_preds: Tensor: num_total_anchor, 4 (dx,dy,dw,dh)
        :param anchors: Tensor: num_total_anchor, 4 (x1,y1,x2,y2)
        :param labels: Tensor: num_total, ; targets for anchors
        :param bbox_targets:  num_total_anchor,4 (dx,dy,dw,dh); targets for bbox preds
        :param num_total_samples:
        :return:
        """

        loss_cla_all = F.cross_entropy(cla_scores, labels, reduction='none') * label_weights

        # foreground: [0,num_class-1]; background: num_class
        pos_inds = ((labels >= 0) &
                    (labels < self.num_classes-1)).nonzero(as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes-1).nonzero(as_tuple=False).view(-1)
        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cla_all[neg_inds].topk(num_neg_samples)
        loss_cla_pos = loss_cla_all[pos_inds].sum()
        loss_cla_neg = topk_loss_cls_neg.sum()
        loss_cla = (loss_cla_pos + loss_cla_neg) / num_total_samples


        #
        loss_bbox = smooth_l1(bbox_preds, bbox_targets, weight=bbox_weights,avg_factor=num_total_samples)
        # from IPython import embed;
        # embed()
        return loss_cla[None], loss_bbox

    def loss(self, outs, gt):
        loc_results, cla_scores = outs
        gt_bboxes, gt_labels = gt

        multi_level_anchors = self.anchor_generator.grid_anchors(image_size=300)
        batch_size = len(loc_results)

        xyxy_anchors = [xywh_xyxy(multi_level_anchor) for multi_level_anchor in multi_level_anchors]
        anchors_list = [xyxy_anchors for _ in range(batch_size)]

        assert gt_labels != None
        (labels_list, labels_weight_list, bboxes_target_list,
         bboxes_weight_list, num_pos_total, num_neg_total) = \
            self.match_(anchors_list, gt_bboxes, gt_labels, batch_size)
        #
        num_total_samples = num_neg_total + num_pos_total
        all_bbox_targets = torch.cat(bboxes_target_list,-2).view(batch_size,-1,4)
        all_bbox_weights = torch.cat(bboxes_weight_list,-2).view(batch_size,-1,4)
        all_anchors = []

        for i in range(batch_size):
            all_anchors.append(torch.cat(anchors_list[i]))
        all_labels = torch.cat(labels_list, -1).view(batch_size, -1)
        all_label_weights = torch.cat(labels_weight_list,-1).view(batch_size, -1)

        # check NaN and Inf
        assert torch.isfinite(cla_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(loc_results).all().item(), \
            'bbox predications become infinite or NaN!'
        # from IPython import embed;
        # embed()
        loss_cla, loss_bbox = multi_apply(self.loss_single_img, cla_scores, loc_results, all_bbox_weights,
                                          all_anchors, all_labels, all_label_weights,
                                           all_bbox_targets, num_total_samples=num_pos_total)  # list[], list[]

        return (sum(loss_cla) + sum(loss_bbox))[0]




if __name__ == '__main__':
    network = SSDHead()
    print(network)
    network.init_weights()
