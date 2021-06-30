# encoding:utf-8

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ...nn import BasicBlock


__all__ = ['RFNet', 'get_rfnet']
# RFNet: Res Fuse Net


class RFNet(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain'):
        super(RFNet, self).__init__()

        self.base = models.resnet18(pretrained=False)
        if pretrained:
            if backbone == 'resnet18':
                f_path = os.path.abspath(os.path.join(root, 'resnet18-5c106cde.pth'))
            if not os.path.exists(f_path):
                raise FileNotFoundError('the pretrained model can not be found')
            self.base.load_state_dict(torch.load(f_path), strict=False)

        self.dep_base = copy.deepcopy(self.base)
        self.dep_base.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.layer0 = nn.Sequential(self.base.conv1, self.base.bn1, self.base.relu)  # [B, 64, h/2, w/2]
        self.pool1 = self.base.maxpool  # [B, 64, h/4, w/4]
        self.layer1 = self.base.layer1  # [B, 64, h/4, w/4]
        self.layer2 = self.base.layer2  # [B, 128, h/8, w/8]
        self.layer3 = self.base.layer3  # [B, 256, h/16, w/16]
        self.layer4 = self.base.layer4  # [B, 512, h/32, w/32]

        self.d_layer0 = nn.Sequential(self.dep_base.conv1, self.dep_base.bn1, self.dep_base.relu)
        self.d_pool1 = self.dep_base.maxpool
        self.d_layer1 = self.dep_base.layer1
        self.d_layer2 = self.dep_base.layer2
        self.d_layer3 = self.dep_base.layer3
        self.d_layer4 = self.dep_base.layer4

        self.fuse1 = RGBDFuse(64)
        self.fuse2 = RGBDFuse(128)
        self.fuse3 = RGBDFuse(256)
        self.fuse4 = RGBDFuse(512)

        self.up4 = nn.Sequential(BasicBlock(512, 512), BasicBlock(512, 256, upsample=True))
        self.up3 = nn.Sequential(BasicBlock(256, 256), BasicBlock(256, 128, upsample=True))
        self.up2 = nn.Sequential(BasicBlock(128, 128), BasicBlock(128, 64, upsample=True))

        self.level_fuse3 = LevelFuse(256)
        self.level_fuse2 = LevelFuse(128)
        self.level_fuse1 = LevelFuse(64)

        self.out_conv = nn.Sequential(BasicBlock(64, 128), BasicBlock(128, 128),
                                      nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, d):
        _, _, h, w = x.size()
        d = self.d_layer0(d)    # [B, 64, h/2, w/2]
        d1 = self.d_pool1(d)    # [B, 64, h/4, w/4]
        d1 = self.d_layer1(d1)  # [B, 64, h/4, w/4]
        d2 = self.d_layer2(d1)  # [B, 128, h/8, w/8]
        d3 = self.d_layer3(d2)
        d4 = self.d_layer4(d3)

        x = self.layer0(x)    # [B, 64, h/2, w/2]
        l0 = self.fuse0(x, d) # [B, 64, h/2, w/2]

        l1 = self.pool1(l0)      # [B, 64, h/4, w/4]
        l1 = self.layer1(l1)     # [B, 64, h/4, w/4]
        l1 = self.fuse1(l1, d1)  # [B, 64, h/4, w/4]

        l2 = self.layer2(l1)     # [B, 128, h/8, w/8]
        l2 = self.fuse2(l2, d2)  # [B, 128, h/8, w/8]

        l3 = self.layer3(l2)     # [B, 256, h/16, w/16]
        l3 = self.fuse3(l3, d3)  # [B, 256, h/16, w/16]

        l4 = self.layer4(l3)     # [B, 512, h/32, w/32]
        l4 = self.fuse4(l4, d4)  # [B, 512, h/32, w/32]

        y4 = self.up4(l4)              # [B, 256, h/16, w/16]
        y3 = self.level_fuse3(y4, l3)

        y3 = self.up3(y3)              # [B, 128, h/8, w/8]
        y2 = self.level_fuse2(y3, l2)  # [B, 128, h/8, w/8]

        y2 = self.up2(y2)              # [B, 64, h/4, w/4]
        y1 = self.level_fuse1(y2, l1)  # [B, 64, h/4, w/4]

        out = self.out_conv(y1)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out


def get_rfnet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',):
    from ...datasets import datasets
    model = RFNet(datasets[dataset.lower()].NUM_CLASS, backbone, pretrained, root=root, )
    return model


class RGBDFuse(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

    def forward(self, x, d):
        return x+d


class LevelFuse(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

    def forward(self, c, x):
        return c+x