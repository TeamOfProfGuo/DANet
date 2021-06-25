# encoding:utf-8

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ...nn import ResidualConvUnit, RefineNetBlock
from ...nn import BasicBlock, FuseBlock, CenterBlock
__all__ = ['MYNet', 'get_mynet']


class MYNet(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 n_features=256, with_att=False, att_type='AG2', ):
        super(MYNet, self).__init__()
        # self.do = nn.Dropout(p=0.5)
        self.base = models.resnet18(pretrained=False)
        if pretrained:
            if backbone == 'resnet18':
                f_path = os.path.abspath(os.path.join(root, 'resnet18-5c106cde.pth'))
            if not os.path.exists(f_path):
                raise FileNotFoundError('the pretrained model can not be found')
            self.base.load_state_dict(torch.load(f_path), strict=False)

        self.dep_base = copy.deepcopy(self.base)
        self.dep_base.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.layer0 = nn.Sequential(self.base.conv1, self.base.bn1, self.base.relu, self.base.maxpool)  # [B, 64, h/4, w/4]
        self.layer1 = self.base.layer1  # [B, 64, h/4, w/4]
        self.layer2 = self.base.layer2  # [B, 128, h/8, w/8]
        self.layer3 = self.base.layer3  # [B, 256, h/16, w/16]
        self.layer4 = self.base.layer4  # [B, 512, h/32, w/32]

        self.d_layer0 = nn.Sequential(self.dep_base.conv1, self.dep_base.bn1, self.dep_base.relu, self.dep_base.maxpool)
        self.d_layer1 = self.dep_base.layer1
        self.d_layer2 = self.dep_base.layer2
        self.d_layer3 = self.dep_base.layer3
        self.d_layer4 = self.dep_base.layer4

        self.fuse1 = RGBDFusionBlock(64, n_features, )
        self.fuse2 = RGBDFusionBlock(128, n_features, )
        self.fuse3 = RGBDFusionBlock(256, n_features, )
        self.fuse4 = RGBDFusionBlock(512, 2*n_features, )

        self.center = CenterBlock(2*n_features, 2*n_features)
        self.up_conv4 = BasicBlock(2*n_features, 2*n_features)
        self.level_fuse3 = FuseBlock([2*n_features, n_features], n_features)
        self.up_conv3 = BasicBlock(n_features, n_features)
        self.level_fuse2 = FuseBlock([n_features, n_features], n_features)
        self.up_conv2 = BasicBlock(n_features, n_features)
        self.level_fuse1 = FuseBlock([n_features, n_features], n_features)
        self.up_conv1 = BasicBlock(n_features, n_features)

        self.out_conv = nn.Sequential(
            BasicBlock(n_features, n_features),
            nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, d):
        _, _, h, w = x.size()
        x = self.layer0(x)  # [B, 64, h/4, w/4]
        l1 = self.layer1(x)   # [B, 64, h/4, w/4]
        l2 = self.layer2(l1)  # [B, 128, h/8, w/8]
        l3 = self.layer3(l2)  # [B, 256, h/16, w/16]
        l4 = self.layer4(l3)  # [B, 512, h/32, w/32]

        d = self.d_layer0(d)
        d1 = self.d_layer1(d)
        d2 = self.d_layer2(d1)
        d3 = self.d_layer3(d2)
        d4 = self.d_layer4(d3)

        l1 = self.fuse1(l1, d1)  # [B, 256, h/4, w/4]
        l2 = self.fuse2(l2, d2)  # [B, 256, h/8, w/8]
        l3 = self.fuse3(l3, d3)  # [B, 256, h/16, w/16]
        l4 = self.fuse4(l4, d4)  # [B, 512, h/32, w/32]

        path4 = self.up_conv4(self.center(l4))              # [B, 512, h/32, w/32]
        path3 = self.up_conv3(self.level_fuse3(path4, l3))  # [B, 256, h/16, w/16]
        path2 = self.up_conv2(self.level_fuse2(path3, l2))  # [B, 256, h/8, w/8]
        path1 = self.up_conv1(self.level_fuse1(path2, l1))  # [B, 256, h/4, w/4]

        out = self.out_conv(path1)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out


def get_mynet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain', n_features=256,
              with_att=False, att_type='AG2'):
    from ...datasets import datasets
    model = MYNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root,
                        n_features=n_features, with_att=with_att, att_type=att_type)
    return model


class RGBDFusionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_rcu=1,):
        super().__init__()

        self.rgb_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.dep_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.rgb_rcu, self.dep_rcu = nn.Sequential(), nn.Sequential()
        for i in range(n_rcu):
            self.rgb_rcu.add_module('rgb_rcu{}'.format(i), ResidualConvUnit(in_ch))
            self.dep_rcu.add_module('dep_rcu{}'.format(i), ResidualConvUnit(in_ch))

        self.rgb_conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.dep_conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, d):
        x = self.rgb_rcu(self.rgb_conv(x))   # [B, in_ch, w, h]
        d = self.dep_rcu(self.dep_conv(d))   # [B, in_ch, w, h]
        x = self.rgb_conv1(x)            # [B, in_ch, w, h]
        d = self.dep_conv1(d)            # [B, in_ch, w, h]

        out = self.relu(x + d)
        return out
