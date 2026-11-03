"""加载训练好的 FCOS 模型，对图片进行路标检测。"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from dataset import CLASS_TO_ID, prepare_sample, read_annotation, read_image
from experiment import PROJECT_DIR, default_checkpoint, load_model, save_json, select_device
from locations import make_locations
from losses import flatten_predictions
from postprocess import postprocess_single_image


@torch.no_grad()
def predict_batch(
    model, images, targets, device, score_threshold=0.3,
    iou_threshold=0.5, max_candidates=1000, max_detections=100,
):
    """预测一个已预处理的批次，逐张返回原图坐标下的检测结果。"""
    model.eval()
    predictions = model(images.to(device))
    flat = flatten_predictions(predictions)
    locations = torch.cat(make_locations(predictions["cls_logits"], model.strides), dim=0)
    results = []
    for index, target in enumerate(targets):
        results.append(postprocess_single_image(
            locations=locations,
            cls_logits=flat["cls_logits"][index],
            box_distances=flat["box_distances"][index],
            center_logits=flat["center_logits"][index],
            resized_size=target["resized_size"],
            original_size=target["original_size"],
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_candidates=max_candidates,
            max_detections=max_detections,
        ))
    return results


def predict_one_image(
    model, image_path, device, score_threshold=0.3,
    iou_threshold=0.5, max_candidates=1000, max_detections=100,
):
    """读取单张图片并预测；不需要 XML 标注。"""
    empty_target = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty(0, dtype=torch.int64),
    }
    image, target = prepare_sample(read_image(image_path), empty_target)
    return predict_batch(
        model, image.unsqueeze(0), [target], device, score_threshold,
        iou_threshold, max_candidates, max_detections,
    )[0]


def save_prediction_image(image_path, result, class_to_id, save_path, target=None):
    """在原图上绘制全部预测框、类别与得分，并保存预览图。"""
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)

    # 可选的绿色真实框仅用于检查，预测本身不读取标注。
    if target is not None:
        for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
            coordinates = [round(value) for value in box]
            draw.rectangle(coordinates, outline="lime", width=3)
            name = next(name for name, value in class_to_id.items() if value == label)
            draw.text((coordinates[0], coordinates[3]), f"GT {name}", fill="lime")

    # 将“类别名称 -> 编号”转换为“编号 -> 类别名称”。
    id_to_class = {
        class_id: name
        for name, class_id in class_to_id.items()
    }

    # 转为 CPU 上的普通列表，供 Pillow 绘图使用。
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    labels = result["labels"].detach().cpu().tolist()

    for box, score, label in zip(boxes, scores, labels):
        # 仅在绘图时取整，result 中的浮点坐标保持原样
        xmin, ymin, xmax, ymax = [round(value) for value in box]

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)

        text = f"{id_to_class[label]} {score:.2f}"
        draw.text(
            (xmin, max(0, ymin - 15)),
            text,
            fill="yellow"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)

    return save_path






def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--image", type=Path, default=PROJECT_DIR / "archive/images/road0.png")
    parser.add_argument("--annotation", type=Path, help="可选 XML，在预览中叠加绿色真实框")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    args = parser.parse_args()
    device = select_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    result = predict_one_image(
        model, args.image, device, args.score_threshold,
        args.nms_iou_threshold, max_detections=args.max_detections,
    )
    target = read_annotation(args.annotation) if args.annotation else None
    output = args.output or PROJECT_DIR / "outputs" / f"{args.image.stem}_pred.png"
    save_prediction_image(args.image, result, CLASS_TO_ID, output, target=target)
    save_json({
        "image": str(args.image), "checkpoint": str(args.checkpoint),
        "epoch": checkpoint["epoch"], "class_to_id": CLASS_TO_ID,
        "score_threshold": args.score_threshold, "nms_iou_threshold": args.nms_iou_threshold,
        **{name: values.detach().cpu().tolist() for name, values in result.items()},
    }, output.with_suffix(".json"))
    print(f"检查点：{args.checkpoint} | 第 {checkpoint['epoch']} 轮 | 设备：{device}")
    print("检测数量：", result["boxes"].shape[0])
    print("预测预览已保存：", output)
    print("完整检测坐标已保存：", output.with_suffix(".json"))


if __name__ == "__main__":
    main()
