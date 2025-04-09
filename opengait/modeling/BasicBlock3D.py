import torch
import torch.nn as nn
from typing import Callable, Optional

def conv3x3x3(in_planes, out_planes, stride=1):
    """3D convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResNet9_3D(nn.Module):
    def __init__(
        self,
        block,
        channels=[64, 128, 256, 512],
        in_channel=1,
        layers=[1, 1, 1, 1],
        strides=[1, (2, 2, 2), (2, 2, 2), 1],  # 👈 支持 tuple
        norm_layer=None
    ):
        super(ResNet9_3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self.inplanes = channels[0]

        self.conv1 = nn.Conv3d(in_channel, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1], norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2], norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3], norm_layer=norm_layer)

    def _make_layer(self, block, planes, blocks, stride, norm_layer):
        # 🧠 自动适配 stride 类型
        if isinstance(stride, int):
            stride = (1, stride, stride)
        elif isinstance(stride, list):
            stride = tuple(stride)

        downsample = None
        if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=(1, 1, 1), norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):  # x: [B, C, T, H, W]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
