"""组合 FCOS 主干、特征金字塔和检测头，完成图像的前向预测。"""

import torch
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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCOS(num_classes=4).to(device)
    model.eval()

    images = torch.randn(2, 3, 416, 416, device=device)

    with torch.no_grad():
        predictions = model(images)

    print("输入设备：", images.device)
    print("输出设备：", predictions["cls_logits"][0].device)
    print("输出尺度数量：", len(predictions["cls_logits"]))
    print("P3 类别得分形状：", predictions["cls_logits"][0].shape)
    print("P7 类别得分形状：", predictions["cls_logits"][-1].shape)

    num_parameters = sum(p.numel() for p in model.parameters())
    print("模型总参数数量：", num_parameters)