import os
import sys
import torch
import torch.nn as nn
import math
#from lib.nn import SynchronizedBatchNorm2d

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


__all__ = ['mobilenet'] # resnet101 is coming soon!

model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

#model_urls = {
#    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
#    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
#    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
#}


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        #SynchronizedBatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        #SynchronizedBatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilate, expand_ratio):
        """
        InvertedResidual: Core block of the MobileNetV2
        :param inp:    (int) Number of the input channels
        :param oup:    (int) Number of the output channels
        :param stride: (int) Stride used in the Conv3x3
        :param dilate: (int) Dilation used in the Conv3x3
        :param expand_ratio: (int) Expand ratio of the Channel Width of the Block
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=dilate, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #SynchronizedBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=oup, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                nn.BatchNorm2d(oup),
                #SynchronizedBatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels=inp, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #SynchronizedBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=dilate, dilation=dilate, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #SynchronizedBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(in_channels=hidden_dim, out_channels=oup, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                nn.BatchNorm2d(oup),
                #SynchronizedBatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, in_size=(448, 896), width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],    # 1/2
            [6, 24, 2, 2, 1],    # 1/4
            [6, 32, 3, 2, 1],    # 1/8
            [6, 64, 4, 1, 2],    # 1/8
            [6, 96, 3, 1, 4],    # 1/8
            [6, 160, 3, 1, 8],   # 1/8
            [6, 320, 1, 1, 16],  # 1/8
        ]

        # building first layer
        assert in_size[0] % 8 == 0
        assert in_size[1] % 8 == 0

        self.input_size = in_size

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0 and s == 2:
                    self.features.append(block(inp=input_channel, oup=output_channel, stride=s, dilate=1, expand_ratio=t))
                else:
                    self.features.append(block(inp=input_channel, oup=output_channel, stride=1, dilate=d, expand_ratio=t))
                input_channel = output_channel



        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x, return_feature_maps=False):
        x = self.features(x)
        #x = x.mean(3).mean(2)
        #x = self.classifier(x)
        if return_feature_maps:
            return [x]
        return [x]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                for i in m.parameters():
                    i.requires_grad = False
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



def mobilenet(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2(n_class=1000)
    state_dict = torch.load('/home/chuwenqing/.encoding/models/mobilenetv2.pth.tar')  # add map_location='cpu' if no gpu
    model.load_state_dict(state_dict)
    return model
