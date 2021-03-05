from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from ...utils import batch_pix_accuracy, batch_intersection_union
from typing import Union, List, Dict, Any, cast


DROPOUT = 0.4

__all__ = ['FuseNet', 'get_fusenet']
class FuseNet(nn.Module):
    def __init__(self, nclass):
        super(FuseNet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(DROPOUT)
        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.cbr1_1 = self.CBR(3, 64)
        self.cbr1_2 = self.CBR(64, 64)
        self.cbr2_1 = self.CBR(64, 128)
        self.cbr2_2 = self.CBR(128, 128)
        self.cbr3_1 = self.CBR(128, 256)
        self.cbr3_2 = self.CBR(256, 256)
        self.cbr3_3 = self.CBR(256, 256)
        self.cbr4_1 = self.CBR(256, 512)
        self.cbr4_2 = self.CBR(512, 512)
        self.cbr4_3 = self.CBR(512, 512)
        self.cbr5_1 = self.CBR(512, 512)
        self.cbr5_2 = self.CBR(512, 512)
        self.cbr5_3 = self.CBR(512, 512)

        self.depth_cbr1_1 = self.CBR(1, 64)
        self.depth_cbr1_2 = self.CBR(64, 64)
        self.depth_cbr2_1 = self.CBR(64, 128)
        self.depth_cbr2_2 = self.CBR(128, 128)
        self.depth_cbr3_1 = self.CBR(128, 256)
        self.depth_cbr3_2 = self.CBR(256, 256)
        self.depth_cbr3_3 = self.CBR(256, 256)
        self.depth_cbr4_1 = self.CBR(256, 512)
        self.depth_cbr4_2 = self.CBR(512, 512)
        self.depth_cbr4_3 = self.CBR(512, 512)
        self.depth_cbr5_1 = self.CBR(512, 512)
        self.depth_cbr5_2 = self.CBR(512, 512)
        self.depth_cbr5_3 = self.CBR(512, 512)

        self.decoder_cbr1_1 = self.CBR(512,512)
        self.decoder_cbr1_2 = self.CBR(512,512)
        self.decoder_cbr1_3 = self.CBR(512,512)
        self.decoder_cbr2_1 = self.CBR(512,256)
        self.decoder_cbr2_2 = self.CBR(256,256)
        self.decoder_cbr2_3 = self.CBR(256,256)
        self.decoder_cbr3_1 = self.CBR(256,128)
        self.decoder_cbr3_2 = self.CBR(128,128)
        self.decoder_cbr4_1 = self.CBR(128,64)
        self.decoder_cbr4_2 = self.CBR(64,64)
        self.decoder_cbr5_1 = self.CBR(64,nclass)

    def load_pretrain(self):
        pass
    
    def forward(self, x):
        x_rgb = x[:,-1:,:,:]
        x_depth = torch.unsqueeze(x[:,-1,:,:],1)

        # cbr1
        x_rgb = self.cbr1_2(self.cbr1_1(x_rgb))
        x_depth = self.depth_cbr1_2(self.depth_cbr1_1(x_depth))
        x_rgb = x_depth + x_rgb                                       # element-wise sum
        x_rgb = self.maxpool(x_rgb)
        x_depth = self.maxpool(x_depth)

        # cbr2
        x_rgb = self.cbr2_2(self.cbr2_1(x_rgb))
        x_depth = self.depth_cbr2_2(self.depth_cbr2_1(x_depth))
        x_rgb = x_depth + x_rgb                                       # element-wise sum
        x_rgb = self.maxpool(x_rgb)
        x_depth = self.maxpool(x_depth)

        # cbr3
        x_rgb = self.cbr3_3(self.cbr3_2(self.cbr3_1(x_rgb)))
        x_depth = self.depth_cbr3_3(self.depth_cbr3_2(self.depth_cbr3_1(x_rgb)))
        x_rgb = x_depth + x_rgb
        x_rgb = self.maxpool(x_rgb)
        x_depth = self.maxpool(x_depth)
        x_rgb = self.dropout(x_rgb)
        x_depth = self.dropout(x_depth)

        # cbr4
        x_rgb = self.cbr4_3(self.cbr4_2(self.cbr4_1(x_rgb)))
        x_depth = self.depth_cbr4_3(self.depth_cbr4_2(self.depth_cbr4_1(x_rgb)))
        x_rgb = x_depth + x_rgb
        x_rgb = self.maxpool(x_rgb)
        x_depth = self.maxpool(x_depth)
        x_rgb = self.dropout(x_rgb)
        x_depth = self.dropout(x_depth)

        # cbr5
        x_rgb = self.cbr5_3(self.cbr5_2(self.cbr5_1(x_rgb)))
        x_depth = self.depth_cbr5_3(self.depth_cbr5_2(self.depth_cbr5_1(x_rgb)))
        x_rgb = x_depth + x_rgb
        x_rgb = self.maxpool(x_rgb)
        x_rgb = self.dropout(x_rgb)

        # decoder_cbr1
        x = self.unpooling(x_rgb)
        x = self.decoder_cbr1_3(self.decoder_cbr1_2(self.decoder_cbr1_1(x)))
        x = self.dropout(x)
        
        # decoder_cbr2
        x = self.unpooling(x)
        x = self.decoder_cbr2_3(self.decoder_cbr2_2(self.decoder_cbr2_1(x)))
        x = self.dropout(x)

        # decoder_cbr3
        x = self.unpooling(x)
        x = self.decoder_cbr3_2(self.decoder_cbr3_1(x))
        x = self.dropout(x)

        # decoder_cbr4
        x = self.unpooling(x)
        x = self.decoder_cbr4_2(self.decoder_cbr4_1(x))

        # decoder_cbr5
        x = self.unpooling(x)
        x = self.decoder_cbr5_1(x)

        return x
    
    def CBR(self, in_channels, out_channels):
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        ]
        return nn.Sequential(*layers)
    def evaluate(self, x, target = None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union



def get_fusenet(dataset="nyu2d", backbone=None, pretrained=False, **kwargs):
    from ...datasets import datasets, acronyms
    model = FuseNet(datasets[dataset.lower()].NUM_CLASS)
    # if pretrained:
    #     from ..model_store import get_model_file
    #     model.load_state_dict(torch.load(
    #         get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model



