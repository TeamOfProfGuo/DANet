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


__all__ = ['DANet_HMD', 'get_danet_hmd']

class DANet_HMD(BaseNet):
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
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANet_HMD, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer)

    def forward(self, image, dep, image_with_dep = None):
        x = image_with_dep if image_with_dep is not None else image
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        ftsize = c4.size()[2:]
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
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        # self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU())

        self.sa = PAM_Module_HMD(inter_channels)
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

class PAM_Module_HMD(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module_HMD, self).__init__()
        self.channel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))
        self.lamb = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

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
        rgb_res = self.softmax(energy)

        # Depth similarity
        d1 = dep.view(m_batchsize, -1, width*height)                         # [B, 1, hw]
        d2 = d1.permute(0, 2, 1)                                             # [B, hw, 1]
        dq = torch.repeat_interleave(d2, width*height, dim = 2)              # [B, hw, hw]
        dk = torch.repeat_interleave(d1, width*height, dim = 1)              # [B, hw, hw]
    
        df = dq - dk                                                         # [B, hw, hw]
        d = torch.mul(df, df)                                                # [B, hw, hw]
        dep_res = self.softmax(d)

        # Finalized simlarity
        simlarity = rgb_res - self.lamb * dep_res
        attention = self.softmax(simlarity)                                                   # [B, hw, hw]

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)                   # [B, 512, hw]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                               # [B, 512, hw]
        out = out.view(m_batchsize, channels, height, width)                                         # [B, 512, h, w]

        out = self.gamma*out + x
        return out


def get_danet_hmd(dataset='pascal_voc', backbone='resnet50', pretrained=False,
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
    model = DANet_HMD(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%ss_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model