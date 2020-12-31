import torch
import torch.nn as nn
from .backbones import VGG
from .heads import SSDHead


class SSD(nn.Module):
    def __init__(self,
                 nms=False,
                 variance=None,
                 num_classes=80):
        super().__init__()
        self.backbone = VGG(depth=16, input_size=300)
        self.head = SSDHead(num_classes=num_classes, nms=nms, variance=variance)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights(pretrained=None)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = SSD()
    # print(model)
    model.init_weights(pretrained='C:/Users/Merlin/Downloads/vgg16_caffe-292e1171.pth')
    input = torch.randn((10, 3, 300, 300))
    for i in range(6):
        print('cls')
        print(model(input)[0][i].shape)
        print('reg')
        print(model(input)[1][i].shape)
