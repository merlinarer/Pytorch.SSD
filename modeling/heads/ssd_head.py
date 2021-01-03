import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modeling.utils.nms import Detect


__all__ = [
    "SSDHead",
]


def conv3x3(in_planes, out_planes, padding=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=padding)


class SSDHead(nn.Module):
    def __init__(self,
                 num_classes=21,
                 nms=False,
                 variance=None,
                 prior=None,
                 anchor_num=(4, 6, 6, 6, 4, 4),
                 in_channels=(512, 1024, 512, 256, 256, 256)):
        super().__init__()
        assert len(anchor_num) == 6
        self.nms = nms
        self.num_classes = num_classes
        self.in_channels = in_channels
        reg_convs = []
        cls_convs = []
        for i, ou in enumerate(anchor_num):
            inplanes = self.in_channels[i]
            reg_convs.append(conv3x3(inplanes, ou * 4, padding=1))
            cls_convs.append(conv3x3(inplanes, ou * self.num_classes, padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes=21,
                             topk=200,
                             conf_thresh=0.01,
                             nms_thresh=0.45,
                             variance=variance)
        self.anchor = Variable(torch.tensor(prior, dtype=torch.float),
                               requires_grad=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(x, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat).permute(0, 2, 3, 1).contiguous())
            bbox_preds.append(reg_conv(feat).permute(0, 2, 3, 1).contiguous())

        if self.nms:
            output = self.detect(torch.cat([o.view(o.size(0), -1, 4) for o in bbox_preds], 1),
                                 self.softmax(torch.cat([o.view(o.size(0), -1, self.num_classes)
                                                         for o in cls_scores], 1)),
                                 self.anchor.cuda()
                                 )
        else:
            output = (
                torch.cat([o.view(o.size(0), -1, 4) for o in bbox_preds], 1),
                torch.cat([o.view(o.size(0), -1, self.num_classes) for o in cls_scores], 1),
                self.anchor.cuda()
            )

        # from IPython import embed
        # embed()

        return output


if __name__ == '__main__':
    network = SSDHead()
    print(network)
    network.init_weights()
