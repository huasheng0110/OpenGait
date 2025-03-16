# 该文件定义了一个简单的 TemporalFusionConv 类，实现了时间维度的下采样
import torch
import torch.nn as nn
import torch.nn.functional as F

class  TemporalFusionConv(nn.Module):
    def __init__(self, in_channels):
        """
        设计一个3D卷积核，在 T 维度上进行融合，并实现时间维度的下采样
        :param in_channels: 输入通道数 C
        """
        super(TemporalFusionConv, self).__init__()
        # 3D卷积: 输入 (B, C, T, W, H) -> 输出 (B, C, T/2, W, H)
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,   # 输入通道 C
            out_channels=in_channels,  # 输出通道保持一致
            kernel_size=(2, 3, 3),     # 在 T 维度上使用 2 进行融合
            stride=(2, 1, 1),          # 在 T 维度上步长为2，实现下采样
            padding=(0, 1, 1),         # 在 W, H 维度上进行 padding，保持空间尺寸
            groups=in_channels         # 逐通道卷积，保持每个通道独立
        )

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (B, T, C, W, H)
        :return: 处理后的张量 (B, T/2, W, H)
        """
        # 调整维度顺序为 (B, C, T, W, H) 以适应 Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        # 调整回 (B, T/2, C, W, H)
        x = x.permute(0, 2, 1, 3, 4)
        return x


class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        """
        进行特征融合：
        1. 先对 (B*T/2, C, W, H) 进行零填充，使其形状与 (B*T, C, W, H) 一致
        2. 通过 2D 卷积融合两个特征
        :param in_channels: 输入通道数 C
        """
        super(FeatureFusion, self).__init__()
        # 2D 卷积用于融合两个特征
        self.fusion_conv = nn.Conv2d(
            in_channels=in_channels * 2,  # 因为输入拼接了两个特征，所以通道数加倍
            out_channels=in_channels,     # 输出通道数与原始特征保持一致
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, original_feat, reduced_feat):
        """
        :param original_feat: 原始特征 (B*T, C, W, H)
        :param reduced_feat: 经过时间融合的特征 (B*T/2, C, W, H)
        :return: 融合后的特征 (B*T, C, W, H)
        """
        B_T, C, W, H = original_feat.shape
        B_T2, _, _, _ = reduced_feat.shape  # B*T/2, C, W, H

        # 计算需要填充的部分
        padding_size = B_T - B_T2
        padded_feat = F.pad(reduced_feat, (0, 0, 0, 0, 0, 0, 0, padding_size))  # 仅在 B 维度补零

        # 维度一致后，拼接两个特征 (B*T, 2C, W, H)
        fused_input = torch.cat([original_feat, padded_feat], dim=1)

        # 通过 2D 卷积进行特征融合
        fused_output = self.fusion_conv(fused_input)

        return fused_output

# # 示例测试
# B, T, C, W, H = 2, 8, 16, 32, 32  # 批次、时间步、通道、宽度、高度
# original_feat = torch.randn(B * T, C, W, H)      # 原始特征 (B*T, C, W, H)
# reduced_feat = torch.randn(B * (T // 2), C, W, H)  # 时间融合后的特征 (B*T/2, C, W, H)

# fusion_model = FeatureFusion(in_channels=C)
# fused_output = fusion_model(original_feat, reduced_feat)

# print("原始特征形状:", original_feat.shape)  # (B*T, C, W, H)
# print("填充后的特征形状:", reduced_feat.shape)  # (B*T/2, C, W, H) -> (B*T, C, W, H)
# print("融合后特征形状:", fused_output.shape)  # 期望 (B*T, C, W, H)



# 示例测试
B, T, C, W, H = 128, 30, 64, 32, 32  # 批次、时间步、通道、宽度、高度
x = torch.randn(B, T, C, W, H)  # 随机输入张量

model = TemporalFusionConv(in_channels=C)
output = model(x)

print("输入形状:", x.shape)  # (B, T, C, W, H)
print("输出形状:", output.shape)  # 期望 (B, T/2, C, W, H)

# 调整x.shape为(B*T, C, W, H)
x = x.view(B * T, C, W, H)
# 调整output.shape为(B*T/2, C, W, H)
output = output.reshape(B * (T // 2), C, W, H)
fusion_model = FeatureFusion(in_channels=C)
fusion_output = fusion_model(x, output)
print("原始特征形状:", x.shape)  # (B*T, C, W, H)
print("填充后的特征形状:", output.shape)  # (B*T/2, C, W, H) -> (B*T, C, W, H)
print("融合后特征形状:", fusion_output.shape)  # 期望 (B*T, C, W, H)
result = fusion_output + x
print("特征形状:", result.shape)  # 期望 (B*T, C, W, H)

