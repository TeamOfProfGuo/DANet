import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ...nn import ResidualConvUnit, MultiResolutionFusion, ChainedResidualPool, RefineNetBlock
from ...nn import AttGate2, AttGate3

__all__ = ['ACDNet', 'get_acdnet']


class ACDNet(nn.Module):
    def __init__(self, n_classes=21, backbone='resnet18', pretrained=True, root='./encoding/models/pretrain',
                 n_features=256, with_CRP=False, with_att=False, att_type='AG2'):
        super(ACDNet, self).__init__()
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
        self.m_base = models.resnet18(pretrained=False)

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

        self.m_layer1 = self.m_base.layer1
        self.m_layer2 = self.m_base.layer2
        self.m_layer3 = self.m_base.layer3
        self.m_layer4 = self.m_base.layer4

        # we can change interim channel tp_ch
        self.fuse0 = ACDFusionBlock(64, 64, n_inputs=2, with_att=with_att, att_type=att_type)
        self.fuse1 = ACDFusionBlock(64, 64, n_inputs=3, with_att=with_att, att_type=att_type)
        self.fuse2 = ACDFusionBlock(128, 128, n_inputs=3, with_att=with_att, att_type=att_type)
        self.fuse3 = ACDFusionBlock(256, 256, n_inputs=3, with_att=with_att, att_type=att_type)
        self.fuse4 = ACDFusionBlock(512, 512, n_inputs=3, with_att=with_att, att_type=att_type)

        # change number of channels
        self.layer4_rn = nn.Conv2d(512, 2 * n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(256, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(128, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_rn = nn.Conv2d(64, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refine4 = RefineNetBlock(2*n_features, [(2*n_features, 32)], with_CRP=with_CRP)
        self.refine3 = RefineNetBlock(n_features, [(2*n_features, 32), (n_features, 16)], with_CRP=with_CRP)
        self.refine2 = RefineNetBlock(n_features, [(n_features, 16), (n_features, 8)], with_CRP=with_CRP)
        self.refine1 = RefineNetBlock(n_features, [(n_features, 8), (n_features, 4)], with_CRP=with_CRP)

        self.out_conv = nn.Sequential(
            ResidualConvUnit(n_features), ResidualConvUnit(n_features),
            nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, d):
        _, _, h, w = x.size()
        x = self.layer0(x)    # [B, 64, h/4, w/4]
        d = self.d_layer0(d)  # [B, 64, h/4, w/4]
        m = self.fuse0(x, d)  # [B, 64, h/4, w/4]

        l1 = self.layer1(x)          # [B, 64, h/4, w/4]
        d1 = self.d_layer1(d)        # [B, 64, h/4, w/4]
        m1 = self.m_layer1(m)        # [B, 64, h/4, w/4]
        m1 = self.fuse1(l1, d1, m1)  # [B, 64, h/4, w/4]

        l2 = self.layer2(l1)         # [B, 128, h/8, w/8]
        d2 = self.d_layer2(d1)       # [B, 128, h/8, w/8]
        m2 = self.m_layer2(m1)       # [B, 128, h/8, w/8]
        m2 = self.fuse2(l2, d2, m2)  # [B, 128, h/8, w/8]

        l3 = self.layer3(l2)         # [B, 256, h/16, w/16]
        d3 = self.d_layer3(d2)       # [B, 256, h/16, w/16]
        m3 = self.m_layer3(m2)       # [B, 256, h/16, w/16]
        m3 = self.fuse3(l3, d3, m3)  # [B, 256, h/16, w/16]

        l4 = self.layer4(l3)         # [B, 512, h/32, w/32]
        d4 = self.d_layer4(d3)       # [B, 512, h/32, w/32]
        m4 = self.m_layer4(m3)       # [B, 512, h/32, w/32]
        m4 = self.fuse4(l4, d4, m4)  # [B, 512, h/32, w/32]

        # change number of channels to match with the input for RefineNet 
        m4 = self.layer4_rn(m4)  # [B, 2*n_feat, h/32, w/32]
        m3 = self.layer3_rn(m3)  # [B, n_feat, h/16, w/16]
        m2 = self.layer2_rn(m2)  # [B, n_feat, h/8, w/8]
        m1 = self.layer1_rn(m1)  # [B, n_feat, h/4, w/4]

        path4 = self.refine4(m4)          # [B, 2*n_feat, h/32, w/32]
        path3 = self.refine3(path4, m3)   # [B, n_feat, h/16, w/16]
        path2 = self.refine2(path3, m2)   # [B, n_feat, h/8, w/8]
        path1 = self.refine1(path2, m1)   # [B, n_feat, h/4, w/4]

        out = self.out_conv(path1)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out


def get_acdnet(dataset='nyud', backbone='resnet18', pretrained=True, root='./encoding/models/pretrain', n_features=256,
                with_CRP=False, with_att=False, att_type='AG2'):
    from ...datasets import datasets
    model = ACDNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, pretrained=pretrained, root=root,
                      n_features=n_features, with_CRP=with_CRP, with_att=with_att, att_type=att_type)
    return model


class ACDFusionBlock(nn.Module):
    def __init__(self, in_ch, tp_ch, n_inputs=2, with_att=False, att_type='AG2'):
        super().__init__()
        self.with_att = with_att
        self.n_inputs = n_inputs
        G = 32   # num of conv groups

        input_lists = ['rgb', 'dep'] if n_inputs==2 else ['rgb', 'dep', 'mrg']
        for feat in input_lists:
            self.add_module('{}_conv'.format(feat),
                            nn.Sequential(
                                nn.Conv2d(in_ch, tp_ch, kernel_size=3, stride=1, padding=1, groups=G, bias=False),
                                nn.BatchNorm2d(tp_ch),
                                nn.ReLU(inplace=True))
                            )

        # fuse with attention
        if with_att:
            if att_type == 'AG2':
                self.att_module = AttGate2(in_ch=tp_ch, M=self.n_inputs, r=16)
            elif att_type == 'AG3':
                self.att_module = AttGate3(in_ch=tp_ch, M=self.n_inputs, r=16)

        self.out_conv = nn.Sequential(nn.Conv2d(tp_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.ReLU(inplace=True)
                                      )

    def forward(self, x, d, m=None):
        x = self.rgb_conv(x)   # [B, tp_ch, w, h]
        d = self.dep_conv(d)   # [B, tp_ch, w, h]
        if self.n_inputs == 3:
            m = self.mrg_conv(m)  # [B, tp_ch, w, h]

        if self.with_att:
            out = self.att_module(x, d) if self.n_inputs == 2 else self.att_module(x, d, m)   # [B, tp_ch, w, h]
        else:
            out = x + d if self.n_inputs == 2 else x+d+m

        return self.out_conv(out)     # [B, in_ch, h, w]
