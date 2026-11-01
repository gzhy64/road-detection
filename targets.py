"""根据真实标注，为 FCOS 预测位置构建训练目标。"""

import torch


# 按 P3～P7 的顺序设置回归距离范围。
REGRESSION_RANGES = (
    (0, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, float("inf")),
)


def get_valid_location_mask(locations, resized_size):
    """判断参考点是否在补边前的有效区域内，返回 [M]。"""
    height, width = resized_size

    xs = locations[:, 0]
    ys = locations[:, 1]

    return ((xs >= 0) & (xs < width) & (ys >= 0) & (ys < height))


def compute_box_distances(locations, boxes):
    """计算每个点相对每个框的四个距离，返回 [M, N, 4]。"""
    # M 个参考点的横、纵坐标，分别整理为 [M, 1]。
    xs = locations[:, 0].unsqueeze(1)
    ys = locations[:, 1].unsqueeze(1)

    # 框边界整理为 [1, N]，通过广播计算所有点与框的组合。
    left = xs - boxes[:, 0].unsqueeze(0)
    top = ys - boxes[:, 1].unsqueeze(0)
    right = boxes[:, 2].unsqueeze(0) - xs
    bottom = boxes[:, 3].unsqueeze(0) - ys

    return torch.stack([left, top, right, bottom], dim=-1)


def get_candidate_mask(box_distances, regression_range):
    """同时满足框内条件和距离范围，返回 [M, N] 候选匹配。"""
    inside_boxes = (box_distances > 0).all(dim=-1)

    # 每个点对每个框，取四个回归距离的最大值。
    max_distances = box_distances.amax(dim=-1)

    lower, upper = regression_range
    in_range = ((max_distances >= lower) & (max_distances <= upper))

    return inside_boxes & in_range


def match_boxes_by_area(boxes, candidate_mask):
    """为每个参考点选择面积最小的候选框，未匹配返回 -1。"""
    num_locations = candidate_mask.shape[0]

    # 没有真实框时，所有参考点都没有匹配目标。
    if boxes.shape[0] == 0:
        return torch.full((num_locations,), -1, dtype=torch.int64, device=candidate_mask.device)

    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    # 候选框保留面积，其余位置设为正无穷。
    candidate_areas = torch.where(candidate_mask, areas.unsqueeze(0), float("inf"))

    min_areas, matched_indices = candidate_areas.min(dim=1)

    # 一整行都是正无穷，说明这个点没有候选框。
    matched_indices[torch.isinf(min_areas)] = -1

    return matched_indices


def compute_centerness(box_distances):
    """根据正样本的 [K,4] 回归距离计算中心度，返回 [K]。"""
    left = box_distances[:, 0]
    top = box_distances[:, 1]
    right = box_distances[:, 2]
    bottom = box_distances[:, 3]

    horizontal_ratio = (torch.minimum(left, right) / torch.maximum(left, right))
    vertical_ratio = (torch.minimum(top, bottom) / torch.maximum(top, bottom))

    return torch.sqrt(horizontal_ratio * vertical_ratio)


def build_targets_per_level(locations, target, regression_range, num_classes=4):
    """为一张图片的一个尺度生成分类、边框回归和中心度目标。"""
    boxes = target["boxes"]
    labels = target["labels"]

    box_distances = compute_box_distances(locations, boxes)
    candidate_mask = get_candidate_mask(box_distances, regression_range)
    valid_mask = get_valid_location_mask(locations, target["resized_size"])

    # 有效位置才允许匹配真实框。
    candidate_mask = candidate_mask & valid_mask.unsqueeze(1)
    matched_indices = match_boxes_by_area(boxes, candidate_mask)

    num_locations = locations.shape[0]

    # 默认所有参考点都是背景。
    cls_targets = torch.full(
        (num_locations,),
        num_classes,
        dtype=torch.int64,
        device=locations.device,
    )

    cls_targets[~valid_mask] = -1

    # 背景位置的回归目标先用零占位。
    reg_targets = torch.zeros(
        (num_locations, 4),
        dtype=locations.dtype,
        device=locations.device,
    )

    positive_mask = matched_indices >= 0

    point_indices = torch.arange(num_locations, device=locations.device)
    positive_indices = point_indices[positive_mask]
    gt_indices = matched_indices[positive_mask]

    # 只为匹配成功的参考点填入真实类别和距离。
    cls_targets[positive_indices] = labels[gt_indices]
    reg_targets[positive_indices] = box_distances[positive_indices, gt_indices]

    # 为所有位置准备中心度占位值，只填入正样本的目标。
    center_targets = torch.zeros(
        (num_locations,),
        dtype=locations.dtype,
        device=locations.device,
    )

    center_targets[positive_indices] = compute_centerness(reg_targets[positive_indices])

    return {
        "labels": cls_targets,
        "box_distances": reg_targets,
        "centerness": center_targets,
    }


def build_targets(locations, targets, num_classes=4):
    """为整个图片批次生成 P3～P7 的训练目标。"""
    if len(locations) != len(REGRESSION_RANGES):
        raise ValueError("参考点尺度数量必须与回归范围数量一致。")

    batch_targets = {
        "labels": [],
        "box_distances": [],
        "centerness": []
    }

    # 外层按尺度处理，内层按图片处理。
    for level_locations, regression_range in zip(locations, REGRESSION_RANGES):
        level_labels = []
        level_reg = []
        level_center = []

        for target in targets:
            assigned = build_targets_per_level(level_locations, target, regression_range, num_classes=num_classes)

            level_labels.append(assigned["labels"])
            level_reg.append(assigned["box_distances"])
            level_center.append(assigned["centerness"])

        batch_targets["labels"].append(torch.stack(level_labels, dim=0))
        batch_targets["box_distances"].append(torch.stack(level_reg, dim=0))
        batch_targets["centerness"].append(torch.stack(level_center, dim=0))

    return batch_targets



if __name__ == '__main__':
    from pathlib import Path

    from torch.utils.data import DataLoader

    from dataset import RoadSignDataset, detection_collate_fn
    from locations import make_locations
    from model import FCOS

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
    model.eval()

    with torch.no_grad():
        predictions = model(images)

    locations = make_locations(predictions["cls_logits"], model.strides)
    batch_targets = build_targets(locations, targets, num_classes=num_classes)

    print("图片批次形状：", images.shape)

    for index in range(len(locations)):
        labels = batch_targets["labels"][index]
        reg = batch_targets["box_distances"][index]
        center = batch_targets["centerness"][index]

        print(
            f"P{index + 3} 训练目标形状：",
            labels.shape,
            reg.shape,
            center.shape,
        )

    all_labels = torch.cat(batch_targets["labels"], dim=1)

    positive_mask = ((all_labels >= 0) & (all_labels < num_classes))
    background_mask = all_labels == num_classes
    ignore_mask = all_labels == -1

    print("合并后分类目标形状：", all_labels.shape)
    print("每张图正样本数：", positive_mask.sum(dim=1))
    print("每张图背景位置数：", background_mask.sum(dim=1))
    print("每张图忽略位置数：", ignore_mask.sum(dim=1))


