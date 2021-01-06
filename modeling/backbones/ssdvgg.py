import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vggmodel
from .build import BACKBONE_REGISTRY

__all__ = [
    "SSDVGG",
]


@BACKBONE_REGISTRY.register()
class SSDVGG(nn.Module):
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.MODEL.BACKBONE.DEPTH
        self.input_size = cfg.INPUT.SIZE_TRAIN
        self.out_feature_indices = cfg.MODEL.BACKBONE.OUT_INDICES

        assert self.depth == 16, 'Only VGG 16 is supported now!'
        assert self.input_size in [300, 512]

        self.features = vggmodel.vgg16(pretrained=True).features
        for m in self.features.modules():
            if isinstance(m, nn.MaxPool2d):
                m.ceil_mode = True
        del self.features[-1]
        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))

        self.extra_inplanes = 1024
        self.extra = self.make_extra_layers(self.extra_inplanes,
                                            self.extra_setting[self.input_size])
        self.l2_norm = L2Norm()
        self.init_weights()

    def init_weights(self):
        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)
        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                outs.append(x)
        outs[0] = self.l2_norm(outs[0])

        return tuple(outs)

    def make_extra_layers(self, inplanes, settings):
        num_layers = 0
        kernel_sizes = [1, 3]
        flag = 0
        layers = []
        for plane in settings:
            k = kernel_sizes[num_layers % 2]
            outplane = plane
            if plane == 'S':
                flag = 1
                continue
            if flag == 1:
                stride = 2
                padding = 1
                flag = 0
            else:
                stride = 1
                padding = 0
            conv = nn.Conv2d(
                inplanes, outplane, k, stride=stride, padding=padding)
            layers.append(conv)
            inplanes = outplane
            num_layers += 1

        if self.input_size == 512:
            layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))

        return nn.Sequential(*layers)


class L2Norm(nn.Module):
    def __init__(self, n_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


if __name__ == '__main__':
    network = SSDVGG(depth=16)
    print(network)
    dict_name = list(network.state_dict())
    # for i, p in enumerate(dict_name):
    #     print(i, p)
    network.init_weights()
