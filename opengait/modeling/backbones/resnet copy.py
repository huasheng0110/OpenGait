from torch.nn import functional as F
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ..modules import BasicConv2d
from ..modules import TemporalFusionConv, FeatureFusion
from ..BasicBlock3D import BasicBlock3D

block_map = {
    'BasicBlock3D': BasicBlock3D,
    'BasicBlock': BasicBlock,
    'Bottleneck': Bottleneck
}
if isinstance(block, str):
    block = block_map[block]

class ResNet9(ResNet):
    def __init__(self, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(ResNet9, self).__init__(block, layers)

        # Not used #
        self.fc = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0], dilate=False)

        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=strides[1], dilate=False)
        # 在layer2后添加一个时间降维的分支，用TemporalFusionConv模块
        self.temporal_fusion_conv = TemporalFusionConv(in_channels=channels[1])
        # 特征融合模块
        self.feature_fusion = FeatureFusion(in_channels=channels[1])

        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)

        x = self.layer1(x)
        x_layer2 = self.layer2(x) # 保存layer2的输出
        B_T, C, W, H = x_layer2.shape
        B = B_T // 30  # T=30
        T = 30
        x_t = x_layer2.view(B, T, C, W, H)
        # 经过时间降维
        x_t = self.temporal_fusion_conv(x_t) # 维度变为 (B, T/2, C, W, H)

        # 变回 (B*T/2, C, W, H)
        x_t = x_t.view(B * (T // 2), C, W, H)  #这里有可能需要用到reshape而不是view

        # 特征融合
        x_fused = self.feature_fusion(x_layer2, x_t)

        # 残差连接
        x_residual = x_layer2 + x_fused

        # x = self.layer2(x)
        x = self.layer3(x_residual)
        x = self.layer4(x)
        return x

# """··································································································"""
# """ResNet9_3D"""
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
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
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
        print("block模块",block)
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
