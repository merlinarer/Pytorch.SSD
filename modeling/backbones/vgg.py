import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG
from .build import BACKBONE_REGISTRY

__all__ = [
    "VGG",
]


def conv3x3(in_planes, out_planes, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=dilation,
        dilation=dilation)


def make_vgg_layer(inplanes,
                   planes,
                   num_blocks,
                   dilation=1,
                   with_bn=False,
                   ceil_mode=True):
    layers = []
    for _ in range(num_blocks):
        layers.append(conv3x3(inplanes, planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        inplanes = planes
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers


@BACKBONE_REGISTRY.register()
class VGG(nn.Module):
    setting = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self,
                 depth=16,
                 input_size=300,
                 out_feature_indices=(22, 34)):
        super().__init__()
        self.out_feature_indices = out_feature_indices
        self.depth = depth
        self.out_feature_indices = out_feature_indices
        self.arc = self.setting[self.depth]
        assert input_size in [300, 512]
        self.input_size = input_size
        layer = []
        inplanes = 3
        for i, num_blocks in enumerate(self.arc):
            planes = 64 * 2 ** i if i < 4 else 512
            tem_layer = make_vgg_layer(
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks
            )
            inplanes = planes
            layer.extend(tem_layer)
        layer.pop(-1)
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*layer))
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
        self.out_feature_indices = out_feature_indices

        self.extra_inplanes = 1024
        self.extra = self.make_extra_layers(self.extra_inplanes,
                                            self.extra_setting[self.input_size])
        self.l2_norm = L2Norm()

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
            self.load_state_dict(state_dict, strict=False)
        else:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight.data)
                    m.bias.data.zero_()

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
    network = VGG(depth=16)
    print(network)
    dict_name = list(network.state_dict())
    # for i, p in enumerate(dict_name):
    #     print(i, p)
    network.init_weights(pretrained='C:/Users/Merlin/Downloads/vgg16_caffe-292e1171.pth')
