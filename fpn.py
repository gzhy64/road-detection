"""构建 FCOS 特征金字塔，融合主干输出的多尺度特征。"""

from torch import nn
import torch.nn.functional as F

class FeaturePyramid(nn.Module):
    """融合主干特征，输出 P3～P7 五级特征金字塔。"""

    def __init__(self, in_channels=(128, 256, 512), out_channels=256):
        super().__init__()

        c3_channels, c4_channels, c5_channels = in_channels

        self.lateral3 = nn.Conv2d(c3_channels, out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4_channels, out_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(c5_channels, out_channels, kernel_size=1)

        # 整合融合后的局部特征，保持空间尺寸不变
        self.output3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 继续下采样，生成额外的两个尺度
        self.extra6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.extra7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, features):
        c3, c4, c5 = features

        lateral3 = self.lateral3(c3)
        lateral4 = self.lateral4(c4)
        inner5 = self.lateral5(c5)

        # 将深层特征放大到相邻浅层的尺寸，再逐元素相加。
        inner4 = lateral4 + F.interpolate(
            inner5,
            size=lateral4.shape[-2:],
            mode="nearest"
        )

        inner3 = lateral3 + F.interpolate(
            inner4,
            size=lateral3.shape[-2:],
            mode="nearest"
        )

        p3 = self.output3(inner3)
        p4 = self.output4(inner4)
        p5 = self.output5(inner5)

        p6 = self.extra6(p5)
        p7 = self.extra7(F.relu(p6))
        return p3, p4, p5, p6, p7
