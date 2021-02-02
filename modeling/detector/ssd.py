import torch
import torch.nn as nn
from modeling.utils import anchor
from torch.autograd import Variable
from ..backbones import build_backbone
from ..heads import build_head
from ..losses import build_loss
from .build import MODEL_REGISTRY
from modeling.utils import anchor



@MODEL_REGISTRY.register()
class SSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)
        # self.loss = build_loss(cfg)
        # self.prior = torch.tensor(anchor.AnchorGenerator()(),
        #                           dtype=torch.float)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def loss_fn(self, out, gt):
        return self.head.loss_(out, gt, )


if __name__ == '__main__':
    model = SSD()
    print(model)
    model.init_weights(pretrained='C:/Users/Merlin/Downloads/vgg16_caffe-292e1171.pth')
    # input = torch.randn((10, 3, 300, 300))
    # for i in range(6):
    #     print('cls')
    #     print(model(input)[0][i].shape)
    #     print('reg')
    #     print(model(input)[1][i].shape)
