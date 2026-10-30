"""构建 FCOS 检测头，用于预测路标类别、边框和中心度。"""

import math

import torch
from torch import nn

from backbone import ConvBlock, TinyBackbone
from fpn import FeaturePyramid


def make_tower(channels, num_blocks=4):
    """构建保持通道数和空间尺寸不变的卷积序列。"""
    layers = []

    for _ in range(num_blocks):
        layers.append(ConvBlock(channels, channels, stride=1))

    return nn.Sequential(*layers)


class FCOSHead(nn.Module):
    """从特征图中预测类别、边框距离和中心度。"""

    def __init__(self, channels=256, num_classes=4):
        super().__init__()

        # 分类和回归分别使用独立的特征提取分支。
        self.cls_tower = make_tower(channels)
        self.reg_tower = make_tower(channels)

        self.cls_pred = nn.Conv2d(
            channels, num_classes, kernel_size=3, padding=1
        )

        # 四个通道依次表示左、上、右、下的距离。
        self.box_pred = nn.Conv2d(
            channels, 4, kernel_size=3, padding=1
        )

        self.center_pred = nn.Conv2d(
            channels, 1, kernel_size=3, padding=1
        )

        # P3～P7 各有一个可学习的回归缩放系数。
        self.scales = nn.Parameter(torch.ones(5))

        self._initialize_weights()

    def _initialize_weights(self, prior_prob=0.01):
        """初始化检测头卷积，并设置分类分支的初始偏置。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 使分类偏置经过 sigmoid 后对应给定的先验概率。
        bias_value = math.log(prior_prob / (1.0 - prior_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)


    def forward_single(self, feature, level_index):
        cls_features = self.cls_tower(feature)
        reg_features = self.reg_tower(feature)

        cls_logits = self.cls_pred(cls_features)

        # 先按当前尺度调节回归输出，再转换为正数距离。
        box_logits = self.box_pred(reg_features)
        box_distances = torch.exp(self.scales[level_index] * box_logits)

        # 按之前论文结构图的连接方式，从分类特征预测中心度。
        center_logits = self.center_pred(cls_features)

        return cls_logits, box_distances, center_logits

    def forward(self, features):
        cls_logits = []
        box_distances = []
        center_logits = []

        # 每个尺度都调用同一套检测头参数。
        for level_index, feature in enumerate(features):
            cls, box, center = self.forward_single(feature, level_index)

            cls_logits.append(cls)
            box_distances.append(box)
            center_logits.append(center)

        return {
            "cls_logits": cls_logits,
            "box_distances": box_distances,
            "center_logits": center_logits
        }



if __name__ == '__main__':
    backbone = TinyBackbone()
    fpn = FeaturePyramid()
    head = FCOSHead(channels=256, num_classes=4)

    images = torch.randn(2, 3, 416, 416)

    # 本次只检查前向输出，省去保存反向传播所需的记录。
    with torch.no_grad():
        backbone_features = backbone(images)
        pyramid_features = fpn(backbone_features)
        prediction = head(pyramid_features)

    for index in range(len(pyramid_features)):
        cls = prediction["cls_logits"][index]
        box = prediction["box_distances"][index]
        center = prediction["center_logits"][index]

        print(
            f"P{index + 3}：",
            cls.shape,
            box.shape,
            center.shape,
        )

    print("各尺度缩放系数：", head.scales.detach().tolist())
    print(
        "分类偏置对应的概率：",
        head.cls_pred.bias.detach().sigmoid().tolist(),
    )

    num_parameters = sum(p.numel() for p in head.parameters())
    print("检测头参数数量：", num_parameters)

