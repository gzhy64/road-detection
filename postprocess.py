"""将 FCOS 的预测解码为边框，并完成检测后处理。"""

import torch

from dataset import resize_boxes
from targets import get_valid_location_mask


DEFAULT_NMS_IOU_THRESHOLD = 0.4
CROSS_CLASS_NMS_IOU_THRESHOLD = 0.8


def decode_boxes(locations, box_distances):
    """将单张图片的参考点 [N,2] 和 ltrb 距离 [N,4] 转为 xyxy 框。"""
    left_top = locations - box_distances[:, :2]
    right_bottom = locations + box_distances[:, 2:]
    return torch.cat([left_top, right_bottom], dim=-1)


def filter_candidates(boxes, cls_logits, center_logits, score_threshold=0.3):
    """融合类别与中心度预测，筛选单张图片的候选框。"""
    cls_probs = cls_logits.sigmoid()

    # [N,1]：同一参考点的中心度作用于所有类别。
    center_probs = center_logits.sigmoid().unsqueeze(-1)
    scores = torch.sqrt(cls_probs * center_probs)

    # 每行记录一个过阈值的“参考点编号、类别编号”。
    point_indices, labels = torch.where(scores >= score_threshold)

    return {
        "boxes": boxes[point_indices],
        "scores": scores[point_indices, labels],
        "labels": labels
    }


def box_iou_one_to_many(box, boxes, eps=1e-6):
    """计算一个 xyxy 框 [4] 与多个框 [N,4] 的 IoU，返回 [N]。"""
    # 交集左上角取较大坐标，右下角取较小坐标。
    left_top = torch.maximum(box[:2], boxes[:, :2])
    right_bottom = torch.minimum(box[2:], boxes[:, 2:])

    # 不相交时，交集宽或高按 0 处理。
    intersection_size = (right_bottom - left_top).clamp_min(0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]

    box_size = (box[2:] - box[:2]).clamp_min(0)
    boxes_size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)

    area = box_size[0] * box_size[1]
    areas = boxes_size[:, 0] * boxes_size[:, 1]

    union = area + areas - intersection

    return intersection / union.clamp_min(eps)


def nms_single_class(boxes, scores, iou_threshold=DEFAULT_NMS_IOU_THRESHOLD):
    """对一组框执行 NMS，返回保留框相对于输入的行号。"""
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)

    # order 保存框的行号，按对应得分从高到低排列。
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)

        remaining = order[1:]
        ious = box_iou_one_to_many(boxes[current], boxes[remaining])

        # 继续处理与当前框重叠程度没有超过阈值的候选。
        order = remaining[ious <= iou_threshold]

    return torch.stack(keep)


def nms_by_class(boxes, scores, labels, iou_threshold=DEFAULT_NMS_IOU_THRESHOLD, max_detections=100):
    """先按类别执行 NMS，再去除跨类别的高度重叠框，返回原始行号。"""
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)

    kept_indices = []

    for class_id in labels.unique():
        # 找到当前类别在完整候选列表中的行号。
        class_indices = (labels == class_id).nonzero().squeeze(1)

        local_keep = nms_single_class(boxes[class_indices], scores[class_indices], iou_threshold=iou_threshold)

        # 将类别子集中的行号映射回完整候选列表。
        kept_indices.append(class_indices[local_keep])

    keep = torch.cat(kept_indices)

    # 同一对象可能被预测成不同类别，优先保留高度重叠候选中的高分项。
    order = scores[keep].argsort(descending=True)
    order = keep[order]
    selected = []
    while order.numel() > 0:
        current = order[0]
        selected.append(current)
        remaining = order[1:]
        ious = box_iou_one_to_many(boxes[current], boxes[remaining])
        different_class = labels[remaining] != labels[current]
        duplicates = different_class & (ious > CROSS_CLASS_NMS_IOU_THRESHOLD)
        order = remaining[~duplicates]

    # 先完成去重再限制数量，避免重复候选占掉其他目标的名额。
    return torch.stack(selected)[:max_detections]


def clip_candidates_to_image(candidates, image_size):
    """将候选框裁剪到有效图片区域，并删除没有面积的框。"""
    height, width = image_size
    boxes = candidates["boxes"].clone()

    # 第 0、2 列是 x 坐标，第 1、3 列是 y 坐标。
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)

    # 裁剪后，宽和高都必须大于 0。
    valid_mask = ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))

    clipped = dict(candidates, boxes=boxes)
    return {name: values[valid_mask] for name, values in clipped.items()}


def restore_candidates_to_original(candidates, resized_size, original_size):
    """将检测框从缩放后坐标还原到原图坐标。"""
    boxes = resize_boxes(candidates["boxes"], old_size=resized_size, new_size=original_size)

    return dict(candidates, boxes=boxes)


def postprocess_single_image(
        locations,
        cls_logits,
        box_distances,
        center_logits,
        resized_size,
        original_size,
        score_threshold=0.3,
        iou_threshold=DEFAULT_NMS_IOU_THRESHOLD,
        max_candidates=1000,
        max_detections=100,
):
    """将单张图片按位置排列的预测转换为原图检测结果。"""
    valid_mask = get_valid_location_mask(locations, resized_size)
    valid_mask &= torch.isfinite(box_distances).all(dim=1)
    valid_mask &= torch.isfinite(center_logits)

    boxes = decode_boxes(locations[valid_mask], box_distances[valid_mask])

    candidates = filter_candidates(boxes, cls_logits[valid_mask], center_logits[valid_mask], score_threshold=score_threshold)
    candidates = clip_candidates_to_image(candidates, resized_size)

    # 在 NMS 前限制候选数量，控制逐框比较的计算量。
    top_indices = candidates["scores"].argsort(descending=True)[:max_candidates]

    for name in candidates:
        candidates[name] = candidates[name][top_indices]

    keep_indices = nms_by_class(
        candidates["boxes"],
        candidates["scores"],
        candidates["labels"],
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )

    selected = {name: values[keep_indices] for name, values in candidates.items()}

    return restore_candidates_to_original(
        selected, resized_size=resized_size, original_size=original_size
    )
