import torch
import torch.nn as nn
from src.resnet import conv3x3
from src.block import MFMI, ICEM, IGDM


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class RecNet(nn.Module):
    def __init__(self, in_channels=1, layers=[2, 2, 2, 2], channels=[8, 16, 32, 64]):
        super().__init__()
        self.layer1 = self._make_encoder_layer(MFMI, in_channels, channels[0], layers[0], stride=2)
        self.layer2 = self._make_encoder_layer(MFMI, channels[0], channels[1], layers[1], stride=2)
        self.layer3 = self._make_encoder_layer(MFMI, channels[1], channels[2], layers[2], stride=2, hcm=True)
        self.layer4 = self._make_encoder_layer(MFMI, channels[2], channels[3], layers[3], stride=2, hcm=True)

        self.icem2 = ICEM(channels[0], channels[1])
        self.icem3 = ICEM(channels[1], channels[2])
        self.icem4 = ICEM(channels[2], channels[3])

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.igdm3 = IGDM(channels[2], channels[3])
        self.igdm2 = IGDM(channels[1], channels[2])
        self.igdm1 = IGDM(channels[0], channels[1])

        self.fuse3 = self._make_decoder_layer(ResBlock, channels[3] + channels[2], channels[2], 2)
        self.fuse2 = self._make_decoder_layer(ResBlock, channels[2] + channels[1], channels[1], 2)
        self.fuse1 = self._make_decoder_layer(ResBlock, channels[1] + channels[0], channels[1], 2)

    def _make_encoder_layer(self, block, in_channels, out_channels, blocks=1, stride=1, hcm=False):
        downsample = None

        if stride != 1 or out_channels != in_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResBlock.expansion),
            )

        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, downsample))
        self.inplanes = out_channels * ResBlock.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, hcm=hcm))
        return nn.Sequential(*layers)
    
    def _make_decoder_layer(self, block, in_channels, out_channels, blocks=1, stride=1):
        downsample = None

        if stride != 1 or out_channels != in_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResBlock.expansion),
            )

        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, downsample))
        self.inplanes = out_channels * ResBlock.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        s_x1 = self.layer1[0](x)
        x1 = self.layer1[1](s_x1)

        s_x2 = self.layer2[0](x1)
        x2 = self.layer2[1](self.icem2([s_x1, s_x2]))

        s_x3 = self.layer3[0](x2)
        x3 = self.layer3[1](self.icem3([s_x2, s_x3]))

        s_x4 = self.layer4[0](x3)
        x4 = self.layer4[1](self.icem4([s_x3, s_x4]))
        
        up_x4 = self.up2(x4)
        out3 = self.fuse3(torch.concat([up_x4, self.igdm3([x3, up_x4])], 1))
        up_x3 = self.up2(out3)
        out2 = self.fuse2(torch.concat([up_x3, self.igdm2([x2, up_x3])], 1))
        up_x2 = self.up2(out2)
        out1 = self.fuse1(torch.concat([up_x2, self.igdm1([x1, up_x2])], 1))

        return out1, out2, out3, x4
