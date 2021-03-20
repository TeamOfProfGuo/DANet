###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample,normalize
from ...nn.da_att import RGBD_PAM_Module
from ...nn.da_att import RGBD_CAM_Module
from .base_double_branch import BaseNet


__all__ = ['DDANet', 'get_ddanet']

class DDANet(BaseNet):
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DDANet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DDANetHead(2048, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        x_rgb = x[:,:-1,:,:]
        x_dep = torch.unsqueeze(x[:,-1,:,:],1)
        _, _, c3, x_rgb = self.base_forward(x_rgb, branch="rgb")
        _, _, c3, x_dep = self.base_forward(x_dep, branch="dep")


        x = self.head(x_rgb, x_dep)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        # x[1] = upsample(x[1], imsize, **self._up_kwargs)
        # x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0]]
        # outputs.append(x[1])
        # outputs.append(x[2])
        return tuple(outputs)
        
class DDANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DDANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv5b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv5d = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = RGBD_PAM_Module(inter_channels)
        self.sc = RGBD_CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x_rgb, x_dep):
        feat1_rgb = self.conv5a(x_rgb)
        feat1_dep = self.conv5b(x_dep)
        sa_feat = self.sa(feat1_rgb, feat1_dep)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2_rgb = self.conv5c(x_rgb)
        feat2_dep = self.conv5d(x_dep)
        sc_feat = self.sc(feat2_rgb, feat2_dep)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        return tuple(output)


def get_ddanet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
           root='./encoding/models/pretrain', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
        'nyud': 'ade',
    }
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = DDANet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%ss_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model