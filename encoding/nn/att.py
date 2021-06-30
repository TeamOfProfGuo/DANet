#encoding:utf-8
import torch
import torch.nn as nn
from functools import reduce
from torch.nn import Module, Softmax, Parameter
__all__ = ['AttGate1', 'AttGate2', 'AttGate3']


# class AttGate2(Module):
#     """ Channel attention module"""
#     def __init__(self, in_ch, reduce_rate=16):
#         super(AttGate2, self).__init__()
#         self.global_avg = nn.AdaptiveAvgPool2d(1)
#         fc_ch = max(in_ch//reduce_rate, 32)
#         self.fc = nn.Sequential(nn.Conv2d(in_ch, fc_ch, kernel_size=1, stride=1, bias=False),
#                                 nn.BatchNorm2d(num_features=fc_ch),
#                                 nn.ReLU(inplace=True))
#         self.a_linear = nn.Conv2d(fc_ch, in_ch, kernel_size=1, stride=1)
#         self.b_linear = nn.Conv2d(fc_ch, in_ch, kernel_size=1, stride=1)
#         self.softmax = Softmax(dim=2)
#
#     def forward(self, x, y):
#         """
#         inputs : x : input feature maps( B X C X H X W); y : input feature maps( B X C X H X W)
#         returns : out: [B, c, h, w]; attention [B, c, 1, 1] for both x and y
#         """
#         u = self.global_avg(x + y)                          # [B, c, 1, 1]
#         z = self.fc(u)                                      # [B, d, 1, 1]
#         a_att, b_att = self.a_linear(z), self.b_linear(z)   # [B, c, 1, 1]
#         att = torch.cat((a_att, b_att), dim=2)              # [B, c, 2, 1]
#         att = self.softmax(att)                             # [B, c, 2, 1]
#
#         out = torch.mul(x, att[:, :, 0:1, :]) + torch.mul(y, att[:, :, 1:2, :])
#         return out

class AttGate1(nn.Module):
    def __init__(self, in_ch, r=4):
        """same as the channel attention in SE module"""
        super(AttGate1, self).__init__()
        int_ch = max(in_ch//r, 32)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(in_ch, int_ch, kernel_size=1, stride=1),
                                nn.BatchNorm2d(int_ch),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(int_ch, in_ch, kernel_size=1, stride=1),
                                nn.Sigmoid())

    def forward(self, x):
        att = self.gap(x)
        att = self.fc(att)  # [B, in_c, 1, 1]
        out = att*x
        return out


class AttGate2(nn.Module):
    def __init__(self, in_ch, M=2, r=4, ret_att=False):
        """ Attention as in SKNet (selective kernel)
        Args:
            features/in_ch: input channel dimensionality.
            M: the number of branches.
            r: the ratio for compute d, the length of z.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(AttGate2, self).__init__()
        print('Att in_ch {} type {}, r {} type {}'.format(in_ch, type(in_ch), r, type(r)))
        d = max(int(in_ch / r), 32)
        self.M = M
        self.in_ch = in_ch
        self.ret_att = ret_att
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # to calculate Z
        self.fc = nn.Sequential(nn.Conv2d(in_ch, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        # 各个分支
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, in_ch, kernel_size=1, stride=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):

        U = reduce(lambda x, y: x+y, inputs)
        batch_size = U.size(0)

        S = self.gap(U)  # [B, c, 1, 1]
        Z = self.fc(S)   # [B, d, 1, 1]

        attention_vectors = [fc(Z) for fc in self.fcs]           # M: [B, c, 1, 1]
        attention_vectors = torch.cat(attention_vectors, dim=1)  # [B, Mc, 1, 1]
        attention_vectors = attention_vectors.view(batch_size, self.M, self.in_ch, 1, 1)  # [B, M, c, 1, 1]
        attention_vectors = self.softmax(attention_vectors)      # [B, M, c, 1, 1]

        feats = torch.cat(inputs, dim=1)  # [B, Mc, h, w]
        feats = feats.view(batch_size, self.M, self.in_ch, feats.shape[2], feats.shape[3])  # [B, M, c, h, w]
        feats_V = torch.sum(feats * attention_vectors, dim=1)

        if self.ret_att:
            return feats_V, attention_vectors
        else:
            return feats_V


class AttGate3(nn.Module):
    def __init__(self, in_ch, M=2, r=4):
        # 输入特征的通道数， 2个分支，bottle-net layer的 reduction rate
        super(AttGate3, self).__init__()
        d = max(int(in_ch*2 / r), 32)
        self.M = M
        self.in_ch = in_ch
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # to calculate Z
        self.fc1 = nn.Sequential(nn.Conv2d(in_ch*2, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, in_ch*2, kernel_size=1, stride=1, bias=False)

        # to calculate attention score
        self.softmax = nn.Softmax(dim=1)

    def forward(self, *inputs):
        # Note: before passed to AttentionModule, x,y has already been preprocessed by conv+BN+ReLU
        x, y = inputs[0], inputs[1]  # [B, c, h, w]
        batch_size = x.size(0)

        u_x = self.gap(x)   # [B, c, 1, 1]
        u_y = self.gap(y)   # [B, c, 1, 1]
        u = torch.cat((u_x, u_y), dim=1)  # [B, 2c, 1, 1]

        z = self.fc1(u)  # [B, d, 1, 1]
        z = self.fc2(z)  # [B, 2c, 1, 1]
        z = z.view(batch_size, 2, self.in_ch, 1, 1)  # [B, 2, c, 1, 1]
        att_score = self.softmax(z)                  # [B, 2, c, 1, 1]

        feats = torch.cat((x,y), dim=1)  # [B, 2c, h, w]
        feats = feats.view(batch_size, 2, self.in_ch, feats.shape[2], feats.shape[3])  # [B, 2, c, h, w]
        feats_V = torch.sum(feats * att_score, dim=1)  # [B, c, h, w]

        out = feats_V if self.M == 2 else feats_V+inputs[2]
        return out
