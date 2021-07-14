# encoding:utf-8

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ...nn import BasicBlock, AttGate1, AttGate2, AttGate3, AttGate3a, AttGate3b, AttGate4c, AttGate5c, AttGate6, AttGate9
from ...nn import PosAtt0, PosAtt1, PosAtt2, PosAtt3, PosAtt3a, PosAtt3c, PosAtt4, PosAtt4a, PosAtt5, PosAtt6, PosAtt6a
from ...nn import PosAtt7, PosAtt7a, PosAtt7b, PosAtt7d, PosAtt9, PosAtt9a, CMPA1, CMPA1a, CMPA2, CMPA2a
from ...nn import ContextBlock, FPA

# RFUNet: Res Fuse U-Net
__all__ = ['RFUNet', 'get_rfunet']


class RFUNet(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 mmf_att=None, mrf_att=None):
        super(RFUNet, self).__init__()

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

        self.fuse0 = RGBDFuse(64, mmf_att=mmf_att, shape=(240, 240))
        self.fuse1 = RGBDFuse(64, mmf_att=mmf_att, shape=(120, 120))
        self.fuse2 = RGBDFuse(128,mmf_att=mmf_att, shape=(60, 60))
        self.fuse3 = RGBDFuse(256,mmf_att=mmf_att, shape=(30, 30))
        self.fuse4 = RGBDFuse(512,mmf_att=mmf_att, shape=(15, 15))

        self.up4 = nn.Sequential(BasicBlock(512, 512), BasicBlock(512, 256, upsample=True))
        self.up3 = nn.Sequential(BasicBlock(256, 256), BasicBlock(256, 128, upsample=True))
        self.up2 = nn.Sequential(BasicBlock(128, 128), BasicBlock(128, 64, upsample=True))

        self.level_fuse3 = LevelFuse(256, mrf_att=mrf_att)
        self.level_fuse2 = LevelFuse(128, mrf_att=mrf_att)
        self.level_fuse1 = LevelFuse(64, mrf_att=mrf_att)

        self.out_conv = nn.Sequential(BasicBlock(64, 128, upsample=True), BasicBlock(128, 128),
                                      nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, d):
        _, _, h, w = x.size()
        d = self.d_layer0(d)  # [B, 64, h/2, w/2]
        d1 = self.d_pool1(d)  # [B, 64, h/4, w/4]
        d1 = self.d_layer1(d1)  # [B, 64, h/4, w/4]
        d2 = self.d_layer2(d1)  # [B, 128, h/8, w/8]
        d3 = self.d_layer3(d2)
        d4 = self.d_layer4(d3)

        x = self.layer0(x)  # [B, 64, h/2, w/2]
        l0 = self.fuse0(x, d)  # [B, 64, h/2, w/2]

        l1 = self.pool1(l0)  # [B, 64, h/4, w/4]
        l1 = self.layer1(l1)  # [B, 64, h/4, w/4]
        l1 = self.fuse1(l1, d1)  # [B, 64, h/4, w/4]

        l2 = self.layer2(l1)  # [B, 128, h/8, w/8]
        l2 = self.fuse2(l2, d2)  # [B, 128, h/8, w/8]

        l3 = self.layer3(l2)  # [B, 256, h/16, w/16]
        l3 = self.fuse3(l3, d3)  # [B, 256, h/16, w/16]

        l4 = self.layer4(l3)  # [B, 512, h/32, w/32]
        l4 = self.fuse4(l4, d4)  # [B, 512, h/32, w/32]

        y4 = self.up4(l4)  # [B, 256, h/16, w/16]
        y3 = self.level_fuse3(y4, l3)

        y3 = self.up3(y3)  # [B, 128, h/8, w/8]
        y2 = self.level_fuse2(y3, l2)  # [B, 128, h/8, w/8]

        y2 = self.up2(y2)  # [B, 64, h/4, w/4]
        y1 = self.level_fuse1(y2, l1)  # [B, 64, h/4, w/4]

        out = self.out_conv(y1)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out


def get_rfunet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
               mmf_att=None, mrf_att=None ):
    from ...datasets import datasets
    model = RFUNet(datasets[dataset.lower()].NUM_CLASS, backbone, pretrained, root=root,
                   mmf_att=mmf_att, mrf_att=mrf_att )
    return model


class RGBDFuse(nn.Module):
    def __init__(self, in_ch, mmf_att=None, shape=None):
        super().__init__()

        self.mmf_att = mmf_att

        if self.mmf_att == 'CA0':     # RGB 与 Dep feature先分别通过channel attention re-weight, 然后相加
            self.rgb_att = AttGate1(in_ch=in_ch)
            self.dep_att = AttGate1(in_ch=in_ch)
        elif self.mmf_att == 'CA1':   # RGB 与 Dep feature先concat, 然后通过channel attention re-weight, 然后通过conv降维
            self.att_module = AttGate1(in_ch=in_ch*2)
            self.out_conv = nn.Conv2d(in_ch*2, in_ch, kernel_size=1, stride=1)
        elif self.mmf_att == 'CA2':   # RGB 与 Dep feature直接通过SKNet Attention, 分别计算RGB 与 Dep的channel weight, 然后相加
            self.att_module = AttGate2(in_ch=in_ch)
        elif self.mmf_att == 'CA3':
            self.att_module = AttGate3(in_ch=in_ch)
        elif self.mmf_att == 'CA3a':
            self.att_module = AttGate3a(in_ch=in_ch)
        elif self.mmf_att == 'CA3b':
            self.att_module = AttGate3b(in_ch=in_ch)
        elif self.mmf_att == 'CA4c':
            self.rgb_att = AttGate4c(in_ch=in_ch, shape=shape)
            self.dep_att = AttGate4c(in_ch=in_ch, shape=shape)
        elif self.mmf_att == 'CA5c':
            self.rgb_att = AttGate5c(in_ch=in_ch)
            self.dep_att = AttGate5c(in_ch=in_ch)
        elif self.mmf_att == 'CA6':
            self.att_module = AttGate6(in_ch=in_ch)
        elif self.mmf_att == 'CA9':
            self.att_module = AttGate9(in_ch=in_ch)
        elif self.mmf_att == 'PA0': # 这里被改过了， 本来是a*x + (1-a)*t
            self.att_module = PosAtt0(ch=in_ch)
            self.out_conv = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1)
        elif self.mmf_att == 'PA1':
            self.att_module = PosAtt1(ch=in_ch)
        elif self.mmf_att == 'PA2':
            self.att_module = PosAtt2(in_ch=in_ch)
        elif self.mmf_att == 'PA3':
            self.att_module = PosAtt3()
        elif self.mmf_att == 'PA3a':
            self.att_module = PosAtt3a(in_ch=in_ch)
        elif self.mmf_att == 'PA4':
            self.att_module = PosAtt4(in_ch=in_ch)
        elif self.mmf_att == 'PA4a':
            self.att_module = PosAtt4a(in_ch=in_ch)
        elif self.mmf_att == 'PA5':
            self.att_module = PosAtt5(in_ch=in_ch)
        elif self.mmf_att == 'PA6':
            self.att_module = PosAtt6(in_ch=in_ch)
        elif self.mmf_att == 'PA6a':
            self.att_module = PosAtt6a(in_ch=in_ch)
        elif self.mmf_att == 'PA7':
            self.att_module = PosAtt7(in_ch=in_ch)
        elif self.mmf_att == 'PA7a':
            self.att_module = PosAtt7a(in_ch=in_ch)
        elif self.mmf_att == 'PA7b':
            self.att_module = PosAtt7b(in_ch=in_ch)
        elif self.mmf_att == 'PA7d':
            self.att_module = PosAtt7d(in_ch=in_ch)
        elif self.mmf_att == 'CB':
            self.rgb_att = ContextBlock(in_ch=in_ch)
            self.dep_att = ContextBlock(in_ch=in_ch)
        elif self.mmf_att == 'PA9':
            self.rgb_att = PosAtt9(in_ch=in_ch)
            self.dep_att = PosAtt9(in_ch=in_ch)
        elif self.mmf_att == 'PA9a':
            self.rgb_att = PosAtt9a(in_ch=in_ch)
            self.dep_att = PosAtt9a(in_ch=in_ch)
        elif self.mmf_att == 'PA3c':
            self.rgb_att = PosAtt3c(in_ch=in_ch)
            self.dep_att = PosAtt3c(in_ch=in_ch)
        elif self.mmf_att == 'CMPA1':
            self.att_module = CMPA1(shape=shape)
        elif self.mmf_att == 'CMPA1a':
            self.att_module = CMPA1a(shape=shape)
        elif self.mmf_att == 'CMPA2':
            self.att_module = CMPA2(shape=shape)
        elif self.mmf_att == 'CMPA2a':
            self.att_module = CMPA2a(shape=shape)

        elif self.mmf_att == 'CA6_CB':
            self.att_module = AttGate6(in_ch=in_ch)
            self.att_module1 = ContextBlock(in_ch=in_ch)
        elif self.mmf_att == 'CA6_PA9':
            self.att_module = AttGate6(in_ch=in_ch)
            self.att_module1 = PosAtt9(in_ch=in_ch)
        elif self.mmf_att == 'PA9_CA6':
            self.rgb_att = PosAtt9(in_ch=in_ch)
            self.dep_att = PosAtt9(in_ch=in_ch)
            self.att_module1 = AttGate6(in_ch=in_ch)
        elif self.mmf_att in ['CA6+PA9', 'CA6vPA9']:
            self.att_module = AttGate6(in_ch=in_ch)
            self.rgb_att = PosAtt9(in_ch=in_ch)
            self.dep_att = PosAtt9(in_ch=in_ch)

    def forward(self, x, d):
        batch_size, ch, _, _ = x.size()
        if self.mmf_att in ['CA0', 'CA4c', 'CA5c']:
            out = self.rgb_att(x) + self.dep_att(d)
        elif self.mmf_att == 'CA1':
            inputs = torch.cat((x, d), dim=1)  # [B, 2c, h, w]
            out = self.att_module(inputs)      # [B, 2c, h, w]
            #return self.out_conv(out)          # [B, c, h, w]
            return out[:, :ch] + out[:, ch:]
        elif self.mmf_att in ['CA2', 'CA3', 'CA3a', 'CA3b', 'CA6', 'CA9']:
            out = self.att_module(x, d)      # 'CA6'这里需要注意顺序，rgb在前面，dep在后面，对dep进行reweight
        elif self.mmf_att == 'PA0':  # 这里被改过了， 本来是a*x + (1-a)*t
            d = self.att_module(x, d)
            out = self.out_conv(torch.cat((x,d), dim=1))
        elif self.mmf_att in ['PA0', 'PA1', 'PA2', 'PA3', 'PA3a', 'PA4', 'PA4a', 'PA5', 'PA6', 'PA6a', 'PA7', 'PA7a',
                              'PA7b', 'PA7d', 'CMPA1', 'CMPA1a', 'CMPA2', 'CMPA2a']:
            out = self.att_module(x, d)   # x is rgb, d is dep
        elif self.mmf_att in ['PA9', 'PA9a', 'PA3c', 'CB']:
            out = self.rgb_att(x) + self.dep_att(d)

        elif self.mmf_att == 'CA6_CB':
            out0 = self.att_module(x, d)
            out = self.att_module1(out0)
        elif self.mmf_att == 'CA6_PA9':
            out0 = self.att_module(x, d)
            out = self.att_module1(out0)
        elif self.mmf_att == 'PA9_CA6':
            rgb0 = self.rgb_att(x)
            dep0 = self.dep_att(d)
            out = self.att_module1(rgb0, dep0)
        elif self.mmf_att == 'CA6+PA9':
            out1 = self.rgb_att(x) + self.dep_att(d)
            out2 = self.att_module(x, d)
            out = out1 + out2
        elif self.mmf_att == 'CA6vPA9':
            out1 = self.rgb_att(x) + self.dep_att(d)
            out2 = self.att_module(x, d)
            out = torch.max(out1, out2)
        else:
            out = x + d

        return out


class LevelFuse(nn.Module):
    def __init__(self, in_ch, mrf_att=None):
        super().__init__()
        self.mrf_att = mrf_att
        if mrf_att == 'PA0':
            self.att_module = PosAtt0(ch=in_ch)
            self.out_conv = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1)
        elif mrf_att == 'CA6':
            self.att_module = AttGate6(in_ch=in_ch)

    def forward(self, c, x):
        if self.mrf_att in ['CA6']:
            return self.att_module(c, x) # 注意深层feature在前，浅层feature在后，对浅层feature进行变换
        elif self.mrf_att == 'PA0':
            x = self.att_module(c, x)
            return self.out_conv( torch.cat((c, x), dim=1) )
        else:
            return c + x
