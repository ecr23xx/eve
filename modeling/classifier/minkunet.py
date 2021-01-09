from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from .resnet import ResNetBase
import time
import logging

import torch
import torch.nn as nn
import MinkowskiEngine as ME


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super(BasicConvolutionBlock, self).__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(True))
        nn.init.constant_(self.net[1].bn.weight, 1.0)
        nn.init.constant_(self.net[1].bn.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super(BasicDeconvolutionBlock, self).__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                inc, outc, kernel_size=ks, stride=stride, dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(True))
        nn.init.constant_(self.net[1].bn.weight, 1.0)
        nn.init.constant_(self.net[1].bn.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=ks, dilation=dilation, stride=stride, dimension=D),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(
                outc, outc, kernel_size=ks, dilation=dilation, stride=1, dimension=D),
            ME.MinkowskiBatchNorm(outc))
        nn.init.constant_(self.net[1].bn.weight, 1.0)
        nn.init.constant_(self.net[1].bn.bias, 0.0)
        nn.init.constant_(self.net[4].bn.weight, 1.0)
        nn.init.constant_(self.net[4].bn.bias, 0.0)

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                ME.MinkowskiConvolution(
                    inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
                ME.MinkowskiBatchNorm(outc))
        if len(self.downsample) > 0:
            nn.init.constant_(self.downsample[1].bn.weight, 1.0)
            nn.init.constant_(self.downsample[1].bn.bias, 0.0)

        self.relu = ME.MinkowskiReLU(True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class MinkUNet(nn.Module):
    def __init__(self, cfg):
        super(MinkUNet, self).__init__()
        self.logger = logging.getLogger('eve.' + __name__)

        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(
                cfg.MODEL.DIM_IN, 32, kernel_size=5, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(True))

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(32, 32, ks=2, stride=2, dilation=1),
            ResidualBlock(32, 32, ks=3, stride=1, dilation=1),
            ResidualBlock(32, 32, ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(32, 32, ks=2, stride=2, dilation=1),
            ResidualBlock(32, 64, ks=3, stride=1, dilation=1),
            ResidualBlock(64, 64, ks=3, stride=1, dilation=1)
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(64, 64, ks=2, stride=2, dilation=1),
            ResidualBlock(64, 128, ks=3, stride=1, dilation=1),
            ResidualBlock(128, 128, ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(128, 128, ks=2, stride=2, dilation=1),
            ResidualBlock(128, 256, ks=3, stride=1, dilation=1),
            ResidualBlock(256, 256, ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(256, 256, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(384, 256, ks=3, stride=1, dilation=1),
                ResidualBlock(256, 256, ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(256, 128, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(192, 128, ks=3, stride=1, dilation=1),
                ResidualBlock(128, 128, ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(128, 96, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(128, 96, ks=3, stride=1, dilation=1),
                ResidualBlock(96, 96, ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(96, 96, ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(128, 96, ks=3, stride=1, dilation=1),
                ResidualBlock(96, 96, ks=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.Sequential(
            ME.MinkowskiConvolution(
                96, cfg.MODEL.NUM_CLASSES,
                kernel_size=1, stride=1, dimension=3))

    def forward(self, x, **kwargs):
        x0 = self.stem(x)     # 30504
        x1 = self.stage1(x0)  # 8039
        x2 = self.stage2(x1)  # 2029
        x3 = self.stage3(x2)  # 489
        x4 = self.stage4(x3)  # 119

        y1 = self.up1[0](x4)
        y1 = ME.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = ME.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = ME.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = ME.cat([y4, x0])
        y4 = self.up4[1](y4)

        out = self.classifier(y4)
        if 'mink' in kwargs and kwargs['mink']:
            return out
        else:
            return out.F


# TODO: simplify code
# Below is copied from https://github.com/StanfordVL/MinkowskiEngine/blob/master/examples/minkunet.py


class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, cfg):
        ResNetBase.__init__(self, cfg.MODEL.DIM_IN, cfg.MODEL.NUM_CLASSES, D=3)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7],
            out_channels,
            kernel_size=1,
            has_bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, **kwargs):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat((out, out_b3p8))
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat((out, out_b2p4))
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat((out, out_b1p2))
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat((out, out_p1))
        out = self.block8(out)

        out = self.final(out)

        if 'mink' in kwargs and kwargs['mink']:
            return out
        else:
            return out.F


class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
