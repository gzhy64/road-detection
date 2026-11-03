"""构建 FCOS 主干网络，用于提取图像特征。"""

from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


def make_stage(in_channels, out_channels):
    """先下采样，再在当前分辨率上继续提取特征。"""
    return nn.Sequential(
        ConvBlock(in_channels, out_channels, stride=2),
        ConvBlock(out_channels, out_channels, stride=1),
    )


class TinyBackbone(nn.Module):
    """通过五个阶段提取特征，返回步长为 8、16、32 的特征图。"""

    def __init__(self):
        super().__init__()

        self.stage1 = make_stage(3, 32)
        self.stage2 = make_stage(32, 64)
        self.stage3 = make_stage(64, 128)
        self.stage4 = make_stage(128, 256)
        self.stage5 = make_stage(256, 512)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        # 保留后三个阶段的特征，供后续多尺度融合使用。
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return c3, c4, c5
