
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResidualConvUnit', 'MultiResolutionFusion', 'ChainedResidualPool', 'ChainedResidualPoolImproved', 'RefineNetBlock']


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, shapes):
        # shapes: [(n_features1, scale1), (n_features2, scale2)  ]
        super().__init__()
        _, min_scale = min(shapes, key=lambda x: x[1])

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, scale = shape
            self.scale_factors.append(scale // min_scale)
            self.add_module("resolve{}".format(i),
                            nn.Conv2d(feat, out_feats, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, *xs):

        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = nn.functional.interpolate(output, scale_factor=self.scale_factors[0], mode='bilinear', align_corners=True)

        for i, x in enumerate(xs[1:], 1): # the value for i starts from 1
            current_out = self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                current_out = nn.functional.interpolate(current_out, scale_factor=self.scale_factors[i], mode='bilinear', align_corners=True)
            output += current_out

        return output


class ChainedResidualPool(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module("block{}".format(i),
                            nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                                          nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 4):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class ChainedResidualPoolImproved(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module("block{}".format(i),
                            nn.Sequential(nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 5):
            path = self.__getattr__("block{}".format(i))(path)
            x += path

        return x


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion, chained_residual_pool, shapes, with_CRP=True):
        super().__init__()
        self.with_CRP = with_CRP
        for i, shape in enumerate(shapes):  # [(n_features1, scale1), (n_features2, scale2)]
            feats = shape[0]
            self.add_module("rcu{}".format(i), nn.Sequential(residual_conv_unit(feats), residual_conv_unit(feats)))

        self.mrf = multi_resolution_fusion(features, shapes) if len(shapes) != 1 else None
        if self.with_CRP:
            self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        if self.with_CRP:
            out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, shapes, with_CRP=True):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion, ChainedResidualPool, shapes, with_CRP=with_CRP)