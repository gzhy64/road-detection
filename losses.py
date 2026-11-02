"""计算 FCOS 的分类、边框回归和中心度损失。"""

import torch
import torch.nn.functional as F


def flatten_predictions(predictions):
    """将各尺度预测整理为按参考点排列的批次张量。"""
    cls_flat = []
    box_flat = []
    center_flat = []

    for index, cls_logits in enumerate(predictions["cls_logits"]):
        box_distances = predictions["box_distances"][index]
        center_logits = predictions["center_logits"][index]

        batch_size, num_classes = cls_logits.shape[:2]

        cls_flat.append(cls_logits.permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes))
        box_flat.append(box_distances.permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
        center_flat.append(center_logits.reshape(batch_size, -1))

    return {
        "cls_logits": torch.cat(cls_flat, dim=1),
        "box_distances": torch.cat(box_flat, dim=1),
        "center_logits": torch.cat(center_flat, dim=1),
    }


def sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """计算分类 Focal Loss 总和，忽略标签 -1。"""
    # 保留有效位置，包括正样本和背景。
    valid_mask = labels >= 0
    logits = logits[valid_mask]
    labels = labels[valid_mask]

    # 将类别编号转换为每个类别的 0/1 监督信号。
    num_classes = logits.shape[-1]
    class_ids = torch.arange(num_classes, device=logits.device)
    targets = (labels.unsqueeze(-1) == class_ids).to(dtype=logits.dtype)

    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = logits.sigmoid()

    # 当前真实 0/1 标签对应的概率。
    p_t = ((probabilities * targets) + (1.0 - probabilities) * (1.0 - targets))

    alpha_t =  (alpha * targets) + (1.0 - alpha) * (1.0 - targets)

    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce_loss

    return loss.sum()


def iou_loss(pred_distances, target_distances, eps=1e-6):
    """计算配对的 ltrb 框的 IoU 损失，输入 [K,4]，返回 [K]。"""
    pred_area = ((pred_distances[:, 0] + pred_distances[:, 2]) * (pred_distances[:, 1] + pred_distances[:, 3]))

    target_area = ((target_distances[:, 0] + target_distances[:, 2]) * (target_distances[:, 1] + target_distances[:, 3]))

    # 每个方向取较小距离，得到交集的四个方向范围。
    inter_distances = torch.minimum(pred_distances, target_distances)
    inter_width = inter_distances[:, 0] + inter_distances[:, 2]
    inter_height = inter_distances[:, 1] + inter_distances[:, 3]
    intersection = inter_width * inter_height

    union = pred_area + target_area - intersection

    iou = intersection / union.clamp_min(eps)

    return -torch.log(iou.clamp(min=eps, max=1.0))


def centerness_loss(logits, targets, positive_mask):
    """只计算正样本位置的中心度 BCE 损失总和。"""
    positive_logits = logits[positive_mask]
    positive_targets = targets[positive_mask]

    return F.binary_cross_entropy_with_logits(positive_logits, positive_targets, reduction="sum")


def compute_losses(predictions, batch_targets):
    """对齐批次预测和训练目标，计算三项损失及总损失。"""
    flat = flatten_predictions(predictions)

    # 训练目标与预测都按 P3～P7 的顺序合并。
    labels = torch.cat(batch_targets["labels"], dim=1)
    reg_targets = torch.cat(batch_targets["box_distances"], dim=1)
    center_targets = torch.cat(batch_targets["centerness"], dim=1)

    num_classes = flat["cls_logits"].shape[-1]
    positive_mask = (labels >= 0) & (labels < num_classes)

    # 分类和中心度损失按整个批次的正样本数量归一化。
    num_positive = positive_mask.sum().clamp_min(1)

    loss_cls = sigmoid_focal_loss(flat["cls_logits"], labels) / num_positive

    loss_center = centerness_loss(flat["center_logits"], center_targets, positive_mask) / num_positive

    # 只计算正样本的回归损失，并用真实中心度加权
    per_point_box_loss = iou_loss(flat["box_distances"][positive_mask], reg_targets[positive_mask])
    box_weights = center_targets[positive_mask]

    loss_box = ((per_point_box_loss * box_weights).sum() / box_weights.sum().clamp_min(1e-6))

    loss_total = loss_cls + loss_center + loss_box

    return {
        "loss_cls": loss_cls,
        "loss_box": loss_box,
        "loss_center": loss_center,
        "loss_total": loss_total,
    }


if __name__ == '__main__':
    from pathlib import Path

    from torch.utils.data import DataLoader

    from dataset import RoadSignDataset, detection_collate_fn
    from locations import make_locations
    from model import FCOS
    from targets import build_targets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4

    dataset = RoadSignDataset(Path(__file__).resolve().parent / "archive")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=detection_collate_fn)

    images, targets = next(iter(loader))
    images = images.to(device)

    for target in targets:
        target["boxes"] = target["boxes"].to(device)
        target["labels"] = target["labels"].to(device)

    model = FCOS(num_classes=num_classes).to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # 保存主干第一个参数张量的副本，用于比较更新前后。
    first_parameter = next(model.backbone.parameters())
    parameter_before = first_parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    predictions = model(images)

    locations = make_locations(predictions["cls_logits"], model.strides)
    batch_targets = build_targets(locations, targets, num_classes=num_classes)

    losses = compute_losses(predictions, batch_targets)

    labels = torch.cat(batch_targets["labels"], dim=1)
    positive_mask = (labels >= 0) & (labels < num_classes)
    print("正样本数量：", positive_mask.sum().item())

    for name, value in losses.items():
        print(f"{name}：{value.item():.6f}")

    print(
        "总损失是否有限：",
        torch.isfinite(losses["loss_total"]).item(),
    )

    losses["loss_total"].backward()

    # 检查梯度是否从检测输出一直传回主干。
    backbone_grad = next(model.backbone.parameters()).grad
    print("主干梯度是否存在：", backbone_grad is not None)

    if backbone_grad is not None:
        print("主干梯度是否有限：", torch.isfinite(backbone_grad).all().item())
        print("主干梯度绝对值均值：", backbone_grad.abs().mean().item())

    # 根据刚才 backward 得到的梯度更新模型参数。
    optimizer.step()

    # 比较同一个参数张量更新前后的差异。
    with torch.no_grad():
        parameter_change = (first_parameter - parameter_before).abs().mean().item()

    print("主干首个参数的平均绝对变化：", parameter_change)
    print("参数是否发生变化：", parameter_change > 0)





