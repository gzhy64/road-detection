"""在真实路标图片上计算逐类别 AP50、精确率和召回率。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import CLASS_TO_ID, RoadSignDataset, detection_collate_fn, read_annotation
from experiment import PROJECT_DIR, default_checkpoint, load_model, prepare_splits, save_json, select_device
from postprocess import box_iou_one_to_many
from predict import predict_batch


def match_detections_for_class(result, target, class_id, iou_threshold=0.5):
    """匹配单张图片中指定类别的预测框和真实框，返回逐框统计。"""
    pred_mask = result["labels"] == class_id
    gt_mask = target["labels"] == class_id

    # 评估不需要梯度，统一在 CPU 上完成匹配。
    boxes = result["boxes"][pred_mask].detach().cpu()
    scores = result["scores"][pred_mask].detach().cpu()
    gt_boxes = target["boxes"][gt_mask].detach().cpu()

    # 高分预测优先匹配，边框和得分使用相同的排序。
    order = scores.argsort(descending=True, stable=True)
    boxes = boxes[order]
    scores = scores[order]

    num_gt = gt_boxes.shape[0]

    # 每个预测先按误检处理，匹配成功后再更新。
    tp = torch.zeros_like(scores)
    fp = torch.ones_like(scores)

    # 记录每个真实框是否已经匹配成功。
    matched_gt = torch.zeros(num_gt, dtype=torch.bool)

    if num_gt > 0:
        for pred_index, box in enumerate(boxes):
            ious = box_iou_one_to_many(box, gt_boxes)
            best_iou, best_gt_index = ious.max(dim=0)

            if (
                best_iou.item() >= iou_threshold
                and not matched_gt[best_gt_index].item()
            ):
                tp[pred_index] = 1.0
                fp[pred_index] = 0.0
                matched_gt[best_gt_index] = True

    return {
        "scores": scores,
        "tp": tp,
        "fp": fp,
        "num_gt": num_gt,
    }


def average_precision(recall, precision):
    """对精确率包络按召回率的变化积分，采用 VOC 2010+ 的全点插值。"""
    recall = torch.as_tensor(recall, dtype=torch.float64)
    precision = torch.as_tensor(precision, dtype=torch.float64)
    mrec = torch.cat([torch.zeros(1), recall, torch.ones(1)])
    mpre = torch.cat([torch.zeros(1), precision, torch.zeros(1)])

    # 从右向左取累计最大值，得到单调不增的精确率包络。
    mpre = mpre.flip(0).cummax(dim=0).values.flip(0)
    changes = (mrec[1:] != mrec[:-1]).nonzero().flatten()
    return ((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]).sum().item()


def summarize_class(matches, score_threshold=0.3):
    """跨图片按得分排序计算 AP，并在指定得分阈值下统计 TP/FP/FN。"""
    num_gt = sum(item["num_gt"] for item in matches)
    scores = torch.cat([item["scores"] for item in matches]) if matches else torch.empty(0)
    tp = torch.cat([item["tp"] for item in matches]) if matches else torch.empty(0)
    fp = torch.cat([item["fp"] for item in matches]) if matches else torch.empty(0)
    order = scores.argsort(descending=True, stable=True)
    scores, tp, fp = scores[order], tp[order], fp[order]

    cumulative_tp, cumulative_fp = tp.cumsum(0), fp.cumsum(0)
    precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp_min(1)
    recall = cumulative_tp / max(num_gt, 1)

    # AP 使用低阈值下的完整排序；展示用 P/R 使用固定的较高阈值。
    selected = scores >= score_threshold
    true_positives = int(tp[selected].sum().item())
    false_positives = int(fp[selected].sum().item())
    p = true_positives / max(true_positives + false_positives, 1)
    r = true_positives / num_gt if num_gt else 0.0
    return {
        "num_gt": num_gt,
        "num_predictions": int(selected.sum().item()),
        "tp": true_positives,
        "fp": false_positives,
        "fn": num_gt - true_positives,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "ap50": average_precision(recall, precision) if num_gt else None,
    }


@torch.no_grad()
def evaluate_model(
    model, dataset, indices, device, batch_size=2, num_workers=0,
    score_threshold=0.05, report_threshold=0.3, iou_threshold=0.5,
    nms_iou_threshold=0.5, max_candidates=1000, max_detections=100,
    verbose=False,
):
    """评估指定样本，预测框和 XML 真实框都使用原图坐标。"""
    if not indices:
        raise ValueError("评估子集为空。")
    if not 0 <= score_threshold <= report_threshold <= 1:
        raise ValueError("阈值必须满足 0 <= 候选得分阈值 <= P/R 得分阈值 <= 1。")
    if iou_threshold != 0.5:
        raise ValueError("当前实现报告 AP50，匹配 IoU 阈值固定为 0.5。")
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=detection_collate_fn,
        generator=torch.Generator().manual_seed(0),
    )
    records = {name: [] for name in CLASS_TO_ID}
    image_index = 0
    was_training = model.training
    model.eval()
    try:
        for images, targets in loader:
            results = predict_batch(
                model, images, targets, device, score_threshold=score_threshold,
                iou_threshold=nms_iou_threshold, max_candidates=max_candidates,
                max_detections=max_detections,
            )
            for result in results:
                annotation = dataset.annotation_paths[indices[image_index]]
                target = read_annotation(annotation)
                for name, class_id in CLASS_TO_ID.items():
                    records[name].append(match_detections_for_class(result, target, class_id))
                image_index += 1
            if verbose and (image_index % 20 < len(results) or image_index == len(indices)):
                print(f"评估进度：{image_index}/{len(indices)}", flush=True)
    finally:
        model.train(was_training)

    classes = {name: summarize_class(items, report_threshold) for name, items in records.items()}
    valid_aps = [item["ap50"] for item in classes.values() if item["ap50"] is not None]
    tp = sum(item["tp"] for item in classes.values())
    fp = sum(item["fp"] for item in classes.values())
    fn = sum(item["fn"] for item in classes.values())
    return {
        "num_images": len(indices),
        "map50": sum(valid_aps) / len(valid_aps) if valid_aps else None,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn,
        "classes": classes,
        "protocol": {
            "ap": "VOC 2010+ all-point interpolated AP at IoU >= 0.5",
            "coordinates": "continuous xyxy, original image, no +1",
            "matching": "highest-IoU same-class GT; duplicate matches are FP",
            "score_threshold": score_threshold,
            "report_threshold": report_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "max_candidates": max_candidates,
            "max_detections": max_detections,
            "classes_without_gt": "AP is null and excluded from mAP",
        },
    }


def print_metrics(metrics):
    """打印逐类别结果；AP/P/R 使用百分数表示。"""
    print("类别           真实数    TP    FP    FN       P       R    AP50")
    for name, item in metrics["classes"].items():
        ap = f"{item['ap50'] * 100:7.2f}" if item["ap50"] is not None else "     --"
        print(
            f"{name:13s} {item['num_gt']:6d} {item['tp']:5d} {item['fp']:5d} "
            f"{item['fn']:5d} {item['precision'] * 100:7.2f} "
            f"{item['recall'] * 100:7.2f} {ap}"
        )
    value = metrics["map50"]
    print("mAP50：", f"{value * 100:.2f}%" if value is not None else "无真实目标，无法计算")
    print(f"总体精确率：{metrics['precision']:.4f} | 总体召回率：{metrics['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "archive")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--image", type=Path, help="仅检查数据集中的指定图片，例如 archive/images/road0.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--report-threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="另存路径，默认在权重目录保存 val_metrics.json 或 test_metrics.json")
    args = parser.parse_args()
    device = select_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    dataset = RoadSignDataset(args.data_dir)
    indices, _ = prepare_splits(dataset, checkpoint=checkpoint)
    selected = indices[args.split]
    split_name = args.split
    if args.image:
        image_path = args.image.resolve()
        selected = [
            i for i, annotation in enumerate(dataset.annotation_paths)
            if (dataset.image_dir / f"{annotation.stem}.png").resolve() == image_path
        ]
        if not selected:
            parser.error("指定图片不在 --data-dir 对应的数据集中。")
        split_name = f"single_{args.image.stem}"
    print(f"检查点：{args.checkpoint} | 第 {checkpoint['epoch']} 轮 | 设备：{device}")
    metrics = evaluate_model(
        model, dataset, selected, device, args.batch_size, args.num_workers,
        score_threshold=args.score_threshold, report_threshold=args.report_threshold,
        nms_iou_threshold=args.nms_iou_threshold, verbose=True,
    )
    metrics.update({"split": split_name, "epoch": checkpoint["epoch"],
                    "checkpoint": str(args.checkpoint), "split_seed": checkpoint["split_seed"]})
    output = args.output or args.checkpoint.parent / f"{split_name}_metrics.json"
    save_json(metrics, output)
    print_metrics(metrics)
    print("评估结果已保存：", output)


if __name__ == "__main__":
    main()
