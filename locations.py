"""生成 FCOS 各尺度预测位置在输入图片中的坐标。"""

import torch


def make_locations_per_level(height, width, stride, device):
    """生成单个尺度的坐标，返回 [H * W, 2]，每行为 (x, y)。"""
    shifts_x = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) * stride

    shifts_y = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) * stride

    grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

    locations = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    return locations


def make_locations(cls_logits, strides):
    """按 P3～P7 的顺序生成各尺度的预测位置坐标。"""
    if len(cls_logits) != len(strides):
        raise ValueError("预测层数量必须与步长数量一致。")

    locations = []

    for logits, stride in zip(cls_logits, strides):
        height, width = logits.shape[-2:]

        points = make_locations_per_level(height, width, stride, logits.device)

        locations.append(points)

    return locations
