"""组合 FCOS 主干、特征金字塔和检测头，完成图像的前向预测。"""

from torch import nn

from backbone import TinyBackbone
from fpn import FeaturePyramid
from head import FCOSHead


class FCOS(nn.Module):
    """从图片批次生成五个尺度的检测预测。"""

    def __init__(self, num_classes=4):
        super().__init__()

        self.num_classes = num_classes
        self.strides = (8, 16, 32, 64, 128)

        self.backbone = TinyBackbone()

        self.fpn = FeaturePyramid(in_channels=(128, 256, 512), out_channels=256)

        self.head = FCOSHead(channels=256, num_classes=num_classes)

    def forward(self, images):
        backbone_features = self.backbone(images)
        pyramid_features = self.fpn(backbone_features)
        predictions = self.head(pyramid_features)

        return predictions
