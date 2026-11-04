"""加载 FCOS 权重，对单张、多张或文件夹中的图片进行检测并保存预览。"""

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


def prepare_prediction_image(image_path):
    """读取并预处理图片，用空标注保留原始和缩放尺寸。"""
    empty_target = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty(0, dtype=torch.int64),
    }
    return prepare_sample(read_image(image_path), empty_target)


def predict_one_image(
    model, image_path, device, score_threshold=0.3,
    iou_threshold=0.5, max_candidates=1000, max_detections=100,
):
    """读取单张图片并预测；不需要 XML 标注。"""
    image, target = prepare_prediction_image(image_path)
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


def collect_image_paths(images=None, image_dir=None, limit=None):
    """收集图片并保持固定顺序，文件夹模式只读取当前层的图片文件。"""
    if image_dir is not None:
        if not image_dir.is_dir():
            raise ValueError(f"图片文件夹不存在：{image_dir}")
        suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        paths = sorted(path for path in image_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in suffixes)
    else:
        paths = images or [PROJECT_DIR / "archive/images/road0.png"]
    # 同一文件重复传入时只处理一次。
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit 必须为正数。")
        paths = paths[:limit]
    if not paths:
        raise ValueError("没有找到可处理的图片。")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"图片不存在：{path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=Path, nargs="+", help="指定一张或多张图片")
    inputs.add_argument("--image-dir", type=Path, help="处理文件夹中的图片，不递归子目录")
    annotations = parser.add_mutually_exclusive_group()
    annotations.add_argument("--annotation", type=Path, help="单图模式的 XML，在预览中叠加绿色真实框")
    annotations.add_argument("--annotation-dir", type=Path, help="按图片名匹配同名 XML 的标注文件夹")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path, help="单图模式的预览文件路径")
    outputs.add_argument("--output-dir", type=Path, help="预览和检测 JSON 的保存文件夹")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, help="最多处理前 N 张图片，不指定则处理全部")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size 必须为正数。")
    if not 0 <= args.score_threshold <= 1 or not 0 <= args.nms_iou_threshold <= 1:
        parser.error("得分阈值和 NMS IoU 阈值必须在 [0,1] 内。")
    if args.max_detections < 1:
        parser.error("--max-detections 必须为正数。")
    try:
        image_paths = collect_image_paths(args.image, args.image_dir, args.limit)
    except ValueError as error:
        parser.error(str(error))
    if len(image_paths) > 1 and (args.output or args.annotation):
        parser.error("处理多张图片时，请使用 --output-dir 和可选的 --annotation-dir。")
    if args.annotation_dir and not args.annotation_dir.is_dir():
        parser.error(f"标注文件夹不存在：{args.annotation_dir}")

    directory_mode = args.image_dir is not None or len(image_paths) > 1
    output_dir = args.output_dir or PROJECT_DIR / "outputs" / ("predictions" if directory_mode else "")
    output_paths = [args.output or output_dir / f"{image.stem}_pred.png" for image in image_paths]
    annotation_paths = [
        args.annotation_dir / f"{image.stem}.xml" if args.annotation_dir else args.annotation
        for image in image_paths
    ]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        parser.error("输入图片存在同名文件，会覆盖输出，请分目录运行或修改图片名。")
    for output in output_paths:
        if output.resolve() in image_paths or output.with_suffix(".json").resolve() in image_paths:
            parser.error("输出文件路径不能覆盖输入图片。")
    for annotation in annotation_paths:
        if annotation is not None and not annotation.is_file():
            parser.error(f"标注文件不存在：{annotation}")

    device = select_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"检查点：{args.checkpoint} | 第 {checkpoint['epoch']} 轮 | 设备：{device}", flush=True)
    print(f"图片数量：{len(image_paths)} | 批次大小：{args.batch_size}", flush=True)
    # 只加载一次模型，图片按小批次处理，避免将整个文件夹一次放入显存。
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start:start + args.batch_size]
        samples = [prepare_prediction_image(path) for path in batch_paths]
        images = torch.stack([image for image, _ in samples])
        targets = [target for _, target in samples]
        results = predict_batch(
            model, images, targets, device, args.score_threshold,
            args.nms_iou_threshold, max_detections=args.max_detections,
        )
        for offset, (image_path, result) in enumerate(zip(batch_paths, results)):
            index = start + offset
            annotation = annotation_paths[index]
            target = read_annotation(annotation) if annotation is not None else None
            output = output_paths[index]
            save_prediction_image(image_path, result, CLASS_TO_ID, output, target=target)
            save_json({
                "image": str(image_path), "checkpoint": str(args.checkpoint),
                "epoch": checkpoint["epoch"], "class_to_id": CLASS_TO_ID,
                "score_threshold": args.score_threshold, "nms_iou_threshold": args.nms_iou_threshold,
                **{name: values.detach().cpu().tolist() for name, values in result.items()},
            }, output.with_suffix(".json"))
            print(f"[{index + 1}/{len(image_paths)}] {image_path.name} | "
                  f"检测 {result['boxes'].shape[0]} 个目标 | 保存：{output}", flush=True)
    print("预测完成，每张预览图旁的同名 JSON 保存完整检测坐标。", flush=True)


if __name__ == "__main__":
    main()
