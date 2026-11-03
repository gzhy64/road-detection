"""将 FCOS 的预测解码为边框，并完成检测后处理。"""

import torch

from dataset import resize_boxes
from targets import get_valid_location_mask


def decode_boxes(locations, box_distances):
    """将单张图片的参考点 [N,2] 和 ltrb 距离 [N,4] 转为 xyxy 框。"""
    xs = locations[:, 0]
    ys = locations[:, 1]

    xmin = xs - box_distances[:, 0]
    ymin = ys - box_distances[:, 1]
    xmax = xs + box_distances[:, 2]
    ymax = ys + box_distances[:, 3]

    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def filter_candidates(boxes, cls_logits, center_logits, score_threshold=0.3):
    """融合类别与中心度预测，筛选单张图片的候选框。"""
    cls_probs = cls_logits.sigmoid()

    # [N,1]：同一参考点的中心度作用于所有类别。
    center_probs = center_logits.sigmoid().unsqueeze(-1)
    scores = torch.sqrt(cls_probs * center_probs)

    # 每行记录一个过阈值的“参考点编号、类别编号”。
    candidate_indices = (scores >= score_threshold).nonzero()
    point_indices = candidate_indices[:, 0]
    labels = candidate_indices[:, 1]

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


def nms_single_class(boxes, scores, iou_threshold=0.5):
    """对同一类别的框执行 NMS，返回保留框相对于输入的行号。"""
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


def nms_by_class(boxes, scores, labels, iou_threshold=0.5, max_detections=100):
    """按类别执行 NMS，返回按得分排序后的输入行号。"""
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

    # 合并各类别后，统一按得分排序并限制总数量。
    order = scores[keep].argsort(descending=True)
    keep = keep[order]

    return keep[:max_detections]


def clip_candidates_to_image(candidates, image_size):
    """将候选框裁剪到有效图片区域，并删除没有面积的框。"""
    height, width = image_size
    boxes = candidates["boxes"].clone()

    # 第 0、2 列是 x 坐标，第 1、3 列是 y 坐标。
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)

    # 裁剪后，宽和高都必须大于 0。
    valid_mask = ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))

    return {
        "boxes": boxes[valid_mask],
        "scores": candidates["scores"][valid_mask],
        "labels": candidates["labels"][valid_mask],
    }


def restore_candidates_to_original(candidates, resized_size, original_size):
    """将检测框从缩放后坐标还原到原图坐标。"""
    boxes = resize_boxes(candidates["boxes"], old_size=resized_size, new_size=original_size)

    return {
        "boxes": boxes,
        "scores": candidates["scores"],
        "labels": candidates["labels"],
    }


def postprocess_single_image(
        locations,
        cls_logits,
        box_distances,
        center_logits,
        resized_size,
        original_size,
        score_threshold=0.3,
        iou_threshold=0.5,
        max_candidates=1000,
        max_detections=100,
):
    """将单张图片按位置排列的预测转换为原图检测结果。"""
    valid_mask = get_valid_location_mask(locations, resized_size)

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

    selected = {
        "boxes": candidates["boxes"][keep_indices],
        "scores": candidates["scores"][keep_indices],
        "labels": candidates["labels"][keep_indices],
    }

    return restore_candidates_to_original(
        selected, resized_size=resized_size, original_size=original_size
    )



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    locations = torch.tensor(
        [
            [100.0, 100.0],
            [140.0, 100.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    box_distances = torch.tensor(
        [
            [40.0, 20.0, 60.0, 50.0],
            [20.0, 20.0, 60.0, 50.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    boxes = decode_boxes(locations, box_distances)

    cls_logits = torch.tensor(
        [
            [2.0, -2.0, 0.0, -2.0],
            [-2.0, -2.0, -2.0, 2.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    center_logits = torch.tensor(
        [2.0, 0.0],
        dtype=torch.float32,
        device=device,
    )

    candidates = filter_candidates(
        boxes,
        cls_logits,
        center_logits,
        score_threshold=0.5,
    )

    print("候选边框：", candidates["boxes"])
    print("候选得分：", candidates["scores"])
    print("候选类别：", candidates["labels"])
    print(
        "候选形状：",
        candidates["boxes"].shape,
        candidates["scores"].shape,
        candidates["labels"].shape,
    )
    print("类别类型：", candidates["labels"].dtype)

    # 检查所有得分都低于阈值的情况。
    empty_candidates = filter_candidates(
        boxes, cls_logits, center_logits, score_threshold=0.99
    )
    print("没有候选时的边框形状：", empty_candidates["boxes"].shape)

    comparison_boxes = torch.tensor(
        [
            [60.0, 80.0, 160.0, 150.0],  # 完全相同
            [70.0, 80.0, 170.0, 150.0],  # 高度重叠
            [120.0, 80.0, 200.0, 150.0],  # 部分重叠
            [200.0, 200.0, 240.0, 240.0],  # 没有重叠
        ],
        dtype=torch.float32,
        device=device
    )

    ious = box_iou_one_to_many(boxes[0], comparison_boxes)

    print("逐框 IoU：", ious)
    print("IoU 属性：", ious.shape, ious.dtype, ious.device)

    empty_ious = box_iou_one_to_many(boxes[0], comparison_boxes[:0])
    print("没有待比较框时的形状：", empty_ious.shape)

    # 本例假设 comparison_boxes 中四个框属于同一类别。
    scores = torch.tensor(
        [0.80, 0.95, 0.70, 0.60],
        dtype=torch.float32,
        device=device,
    )

    keep_indices = nms_single_class(
        comparison_boxes, scores, iou_threshold=0.5
    )

    print("NMS 保留的行号：", keep_indices)
    print("NMS 保留的边框：", comparison_boxes[keep_indices])
    print("NMS 保留的得分：", scores[keep_indices])
    print(
        "保留行号的属性：",
        keep_indices.shape,
        keep_indices.dtype,
        keep_indices.device,
    )

    empty_keep = nms_single_class(
        comparison_boxes[:0],
        scores[:0],
    )
    print("空输入保留行号：", empty_keep)

    # 复制第 1 号框作为新候选，给它另一个类别。
    multi_boxes = torch.cat(
        [comparison_boxes, comparison_boxes[1:2]],
        dim=0,
    )

    multi_scores = torch.tensor(
        [0.80, 0.95, 0.70, 0.60, 0.90],
        dtype=torch.float32,
        device=device,
    )

    multi_labels = torch.tensor(
        [0, 0, 0, 0, 2],
        dtype=torch.int64,
        device=device,
    )

    multi_keep = nms_by_class(
        multi_boxes,
        multi_scores,
        multi_labels,
        iou_threshold=0.5,
    )

    print("多类别 NMS 保留行号：", multi_keep)
    print("保留边框：", multi_boxes[multi_keep])
    print("保留得分：", multi_scores[multi_keep])
    print("保留类别：", multi_labels[multi_keep])

    top_two = nms_by_class(
        multi_boxes, multi_scores, multi_labels, max_detections=2
    )
    print("最多保留两个时的行号：", top_two)

    empty_multi_keep = nms_by_class(
        multi_boxes[:0], multi_scores[:0], multi_labels[:0]
    )
    print("多类别空输入结果：", empty_multi_keep)

    boundary_candidates = {
        "boxes": torch.tensor(
            [
                [-10.0, 20.0, 300.0, 450.0],  # 跨越图片边界
                [300.0, 30.0, 350.0, 100.0],  # 完全位于右侧补边区
                [50.0, 60.0, 150.0, 180.0],   # 完全位于有效区域
                [30.0, -50.0, 100.0, -10.0],  # 完全位于图片上方
            ],
            dtype=torch.float32,
            device=device,
        ),
        "scores": torch.tensor(
            [0.9, 0.8, 0.7, 0.6],
            dtype=torch.float32,
            device=device,
        ),
        "labels": torch.tensor(
            [0, 1, 2, 3],
            dtype=torch.int64,
            device=device,
        ),
    }

    clipped_candidates = clip_candidates_to_image(
        boundary_candidates,
        image_size=(416, 278),
    )

    print("裁剪后的边框：", clipped_candidates["boxes"])
    print("保留的得分：", clipped_candidates["scores"])
    print("保留的类别：", clipped_candidates["labels"])

    clipped_empty = clip_candidates_to_image(
        empty_candidates,
        image_size=(416, 278),
    )
    print("空候选裁剪后的形状：", clipped_empty["boxes"].shape)

    resized_size = (416, 278)
    original_size = (400, 267)

    restored_candidates = restore_candidates_to_original(clipped_candidates, resized_size=resized_size, original_size=original_size)

    print("还原后的边框：", restored_candidates["boxes"])
    print("还原后的得分：", restored_candidates["scores"])
    print("还原后的类别：", restored_candidates["labels"])
    print(
        "还原后的边框属性：",
        restored_candidates["boxes"].shape,
        restored_candidates["boxes"].dtype,
        restored_candidates["boxes"].device,
    )

    restored_empty = restore_candidates_to_original(
        clipped_empty,
        resized_size=resized_size,
        original_size=original_size,
    )
    print("空候选还原后的形状：", restored_empty["boxes"].shape)

    test_locations = torch.tensor(
        [[100.0, 100.0], [110.0, 100.0], [300.0, 100.0]],
        dtype=torch.float32,
        device=device,
    )

    test_box_distances = torch.tensor(
        [
            [40.0, 20.0, 60.0, 50.0],
            [40.0, 20.0, 60.0, 50.0],
            [250.0, 20.0, 60.0, 50.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    test_cls_logits = torch.tensor(
        [
            [3.0, -3.0, -3.0, -3.0],
            [2.0, -3.0, -3.0, -3.0],
            [5.0, -5.0, -5.0, -5.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    test_center_logits = torch.tensor(
        [2.0, 2.0, 5.0],
        dtype=torch.float32,
        device=device,
    )

    result = postprocess_single_image(
        test_locations,
        test_cls_logits,
        test_box_distances,
        test_center_logits,
        resized_size=resized_size,
        original_size=original_size,
        score_threshold=0.5,
    )

    print("组合处理后的边框：", result["boxes"])
    print("组合处理后的得分：", result["scores"])
    print("组合处理后的类别：", result["labels"])

    empty_result = postprocess_single_image(
        test_locations,
        test_cls_logits,
        test_box_distances,
        test_center_logits,
        resized_size=resized_size,
        original_size=original_size,
        score_threshold=0.999,
    )

    print(
        "组合处理的空结果形状：",
        empty_result["boxes"].shape,
        empty_result["scores"].shape,
        empty_result["labels"].shape,
    )
