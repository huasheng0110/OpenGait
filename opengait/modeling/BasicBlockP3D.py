import torch
import torch.nn as nn


class BasicBlockP3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1,1,1), downsample=None, norm_layer=None):
        super(BasicBlockP3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        # 分解卷积：先空间卷积，再时间卷积
        self.conv2d = nn.Conv3d(inplanes, planes, kernel_size=(1,3,3),
                                stride=(1, stride[1], stride[2]), padding=(0,1,1), bias=False)
        self.bn2d = norm_layer(planes)

        self.conv1d = nn.Conv3d(planes, planes, kernel_size=(3,1,1),
                                stride=(stride[0], 1, 1), padding=(1,0,0), bias=False)
        self.bn1d = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn2d(self.conv2d(x)))
        out = self.bn1d(self.conv1d(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
