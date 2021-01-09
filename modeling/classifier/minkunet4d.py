import torch
import torch.nn as nn

import MinkowskiEngine as ME


class BasicConvolutionBlock4d(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
        super(BasicConvolutionBlock4d, self).__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, dilation=dilation, stride=stride, dimension=D,
                kernel_generator=ME.KernelGenerator(
                    kernel_size=(ks, ks, ks, 1), dimension=D,
                    region_type=ME.RegionType.HYPERCUBE)),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(True))
        nn.init.constant_(self.net[1].bn.weight, 1.0)
        nn.init.constant_(self.net[1].bn.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class BasicDeconvolutionBlock4d(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, D=3):
        super(BasicDeconvolutionBlock4d, self).__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                inc, outc, stride=stride, dimension=D,
                kernel_generator=ME.KernelGenerator(
                    kernel_size=(ks, ks, ks, 1), dimension=D,
                    region_type=ME.RegionType.HYPERCUBE)),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(True))
        nn.init.constant_(self.net[1].bn.weight, 1.0)
        nn.init.constant_(self.net[1].bn.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class ResidualBlock4d(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=4):
        super(ResidualBlock4d, self).__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, dimension=D,
                kernel_generator=ME.KernelGenerator(
                    kernel_size=ks, dimension=D,
                    region_type=ME.RegionType.HYBRID,
                    axis_types=(ME.RegionType.HYPERCUBE, ME.RegionType.HYPERCUBE,
                                ME.RegionType.HYPERCUBE, ME.RegionType.HYPERCROSS))),
            ME.MinkowskiBatchNorm(outc),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(
                outc, outc, dimension=D,
                kernel_generator=ME.KernelGenerator(
                    kernel_size=ks, dimension=D,
                    region_type=ME.RegionType.HYBRID,
                    axis_types=(ME.RegionType.HYPERCUBE, ME.RegionType.HYPERCUBE,
                                ME.RegionType.HYPERCUBE, ME.RegionType.HYPERCROSS))),
            ME.MinkowskiBatchNorm(outc))
        nn.init.constant_(self.net[1].bn.weight, 1.0)
        nn.init.constant_(self.net[1].bn.bias, 0.0)
        nn.init.constant_(self.net[4].bn.weight, 1.0)
        nn.init.constant_(self.net[4].bn.bias, 0.0)

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else nn.Sequential(
            ME.MinkowskiConvolution(
                inc, outc, kernel_size=1, dilation=1, stride=stride, dimension=D),
            ME.MinkowskiBatchNorm(outc))
        if len(self.downsample) > 0:
            nn.init.constant_(self.downsample[1].bn.weight, 1.0)
            nn.init.constant_(self.downsample[1].bn.bias, 0.0)

        self.relu = ME.MinkowskiReLU(True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class MinkUNet4d(nn.Module):
    def __init__(self, cfg):
        super(MinkUNet4d, self).__init__()

        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(
                3, 32, dimension=4,
                kernel_generator=ME.KernelGenerator(
                    kernel_size=(5, 5, 5, 1), dimension=4,
                    region_type=ME.RegionType.HYPERCUBE)),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(True))
        nn.init.constant_(self.stem[1].bn.weight, 1.0)
        nn.init.constant_(self.stem[1].bn.bias, 0.0)

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock4d(32, 32, ks=2, stride=(2, 2, 2, 1), D=4),
            ResidualBlock4d(32, 32, ks=3),
            ResidualBlock4d(32, 32, ks=3),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock4d(32, 32, ks=2, stride=(2, 2, 2, 1), D=4),
            ResidualBlock4d(32, 64, ks=3),
            ResidualBlock4d(64, 64, ks=3)
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock4d(64, 64, ks=2, stride=(2, 2, 2, 1), D=4),
            ResidualBlock4d(64, 128, ks=3),
            ResidualBlock4d(128, 128, ks=3),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock4d(128, 128, ks=2, stride=(2, 2, 2, 1), D=4),
            ResidualBlock4d(128, 256, ks=3),
            ResidualBlock4d(256, 256, ks=3),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock4d(
                256, 256, ks=2, stride=(2, 2, 2, 1), D=4),
            nn.Sequential(
                ResidualBlock4d(384, 256, ks=3),
                ResidualBlock4d(256, 256, ks=3),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock4d(
                256, 128, ks=2, stride=(2, 2, 2, 1), D=4),
            nn.Sequential(
                ResidualBlock4d(192, 128, ks=3),
                ResidualBlock4d(128, 128, ks=3),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock4d(128, 96, ks=2, stride=(2, 2, 2, 1), D=4),
            nn.Sequential(
                ResidualBlock4d(128, 96, ks=3),
                ResidualBlock4d(96, 96, ks=3),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock4d(96, 96, ks=2, stride=(2, 2, 2, 1), D=4),
            nn.Sequential(
                ResidualBlock4d(128, 96, ks=3),
                ResidualBlock4d(96, 96, ks=3),
            )
        ])

        self.classifier = nn.Sequential(
            ME.MinkowskiConvolution(
                96, cfg.MODEL.NUM_CLASSES,
                kernel_size=1, dimension=4))

    def forward(self, x, **kargs):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

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
        return out
