###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample
import torch.nn.functional as F

from .base import BaseNet
from .fcn import FCNHead

__all__ = ['DEEPLAB', 'get_deeplab', 'get_deeplab_resnet50_pcontext', 'get_deeplab_resnet50_ade']

class DEEPLAB(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = DEEPLAB(nclass=21, backbone='resnet50')
    >>> print(model)
    """
    def __init__(self, nclass, backbone, aux=True, se_loss=False, lateral=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DEEPLAB, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        #elf.head = DEEPLABHead(2048, nclass,lateral=lateral, norm_layer=norm_layer, up_kwargs=self._up_kwargs)
        self.head = DEEPLABHead(1280, nclass, lateral=False, norm_layer=norm_layer, up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)
        x = list(self.head(*features))
        x[0] = F.upsample(x[0], imsize, **self._up_kwargs)

        #_, _, c3, c4 = self.base_forward(x)

        #x = self.head(c4)
        #x = upsample(x, imsize, **self._up_kwargs)
        #outputs = [x]
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = upsample(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        return tuple(x)

class Classifier_Module(nn.Module):
    def __init__(self, in_channels, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out
        
class DEEPLABHead(nn.Module):
    def __init__(self, in_channels, out_channels, lateral=True, norm_layer=None, up_kwargs=None):
        super(DEEPLABHead, self).__init__()
        #inter_channels = in_channels // 4
        self.lateral = lateral
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True))
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
            ])
            self.fusion = nn.Sequential(
                    nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True))


        #self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                           norm_layer(inter_channels),
        #                           nn.ReLU(),
        #                           nn.Dropout2d(0.1, False),
        #                           nn.Conv2d(inter_channels, out_channels, 1))
        self.conv6 = Classifier_Module(512, [6, 12, 18, 24], [6, 12, 18, 24], out_channels)

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        return tuple([self.conv6(feat)])


def get_deeplab(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    r"""FCN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_deeplab(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    #kwargs['lateral'] = True if dataset.lower() == 'pcontext' else False
    kwargs['lateral'] = True
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation, ContextSegmentation, CityscapesSegmentation
    model = DEEPLAB(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('deeplab_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict= False)
    return model

def get_deeplab_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deeplab_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('pcontext', 'resnet50', pretrained, root=root, aux=False, **kwargs)

def get_deeplab_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_deeplab_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('ade20k', 'resnet50', pretrained, root=root, **kwargs)
