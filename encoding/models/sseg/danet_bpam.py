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
from torch.nn import Module, Sequential, Conv2d, Parameter, Softmax
from torch.nn.functional import upsample,normalize
from ...nn.da_att import PAM_Module
from ...nn.da_att import CAM_Module
from .base import BaseNet

GPUS = [0, 1, 2, 3]

__all__ = ['DANet_bpam', 'get_danet_bpam']

class DANet_bpam(BaseNet):
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
    """
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, 
                    dep_resize_method='itpl', dep_order=2, geo_siml=False, **kwargs):
        super(DANet_bpam, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer, dep_order, geo_siml)

        assert dep_resize_method in ('itpl', 'ap')
        self.resize_method = dep_resize_method

    def forward(self, image, dep, image_with_dep = None):
        x = image_with_dep if image_with_dep is not None else image
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        ftsize = c4.size()[2:]
        if self.resize_method == 'itpl':
            resized_dep = nn.functional.interpolate(dep, ftsize, mode='bilinear', align_corners=False)
        else:
            ave_pool = nn.AdaptiveAvgPool2d(ftsize)
            resized_dep = ave_pool(dep)

        x = self.head(c4, resized_dep)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        # x[1] = upsample(x[1], imsize, **self._up_kwargs)
        # x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0]]
        # outputs.append(x[1])
        # outputs.append(x[2])
        return tuple(outputs)
        
class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, dep_order, geo_siml):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        # self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU())

        self.sa = BPAM_Module(inter_channels, dep_order, use_geo_siml=geo_siml)
        # self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        # self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        # self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x, dep):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1, dep)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        output = [sa_output]

        # feat2 = self.conv5c(x)
        # sc_feat = self.sc(feat2)
        # sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        # feat_sum = sa_conv+sc_conv
        
        # sasc_output = self.conv8(feat_sum)

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        return tuple(output)

class BPAM_Module(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN
    def __init__(self, in_dim, dep_order = 2, use_geo_siml = False, **kwargs):
        super(BPAM_Module, self).__init__()
        self.channel_in = in_dim
        self.use_geo_siml = use_geo_siml
        self.dep_order = dep_order

        if use_geo_siml:
            self.load_geo_siml()

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))

        self.lamb1 = Parameter(torch.zeros(1))
        self.lamb2 = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    
    def load_geo_siml(self):
        self.geo_siml = {}
        gs = torch.load('/gpfsnyu/scratch/hl3797/DANet/encoding/models/sseg/gs.pth')['geo_siml']
        for i in GPUS:
            self.geo_siml[i] = gs.to("cuda:" + str(i))

    def forward(self, x, dep):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, channels, height, width = x.size()

        # RGB similarity
        query = self.query_conv(x)                                                    # [B, 64, h, w]
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)       # [B, hw, 64]
        key = self.key_conv(x)                                                        # [B, 64, h, w]
        proj_key = key.view(m_batchsize, -1, width*height)                            # [B, 64, hw]
        energy = torch.bmm(proj_query, proj_key)                                      # [B, hw, hw]
        # rgb_siml = self.softmax(energy)

        # Depth similarity
        d1 = dep.view(m_batchsize, -1, width*height)                                  # [B, 1, hw]
        d2 = d1.permute(0, 2, 1)                                                      # [B, hw, 1]
        d = d2 - d1                                                                   # [B, hw, hw]
        if self.dep_order == 1:
            dep_diss = torch.abs(d)
        elif self.dep_order == 2:
            dep_diss = torch.pow(d, 2)
        else:
            print('[Dep Siml]: Invalid depth order.')
            exit(0)

        # Geo similarity
        if self.use_geo_siml:
            geo_siml = self.geo_siml[torch.cuda.current_device()].expand(m_batchsize, width*height, width*height)

        # Finalized simlarity
        # simlarity = rgb_siml + self.lamb1 * dep_siml + self.lamb2 * geo_siml
        simlarity = energy - (self.lamb1 * dep_diss) + (self.lamb2 * geo_siml if self.use_geo_siml else 0)
        attention = self.softmax(simlarity)                                                   # [B, hw, hw]

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)                   # [B, 512, hw]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                               # [B, 512, hw]
        out = out.view(m_batchsize, channels, height, width)                                         # [B, 512, h, w]

        out = self.gamma * out + x
        return out


def get_danet_bpam(dataset='pascal_voc', backbone='resnet50', pretrained=False,
           root='./encoding/models/pretrain', **kwargs):
    r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
    """
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
    model = DANet_bpam(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%ss_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model