"""加载训练好的 FCOS 模型，对图片进行路标检测。"""

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from dataset import read_image, resize_image, pad_image
from locations import make_locations
from losses import flatten_predictions
from model import FCOS
from postprocess import postprocess_single_image


def predict_one_image(model, image_path, device, score_threshold=0.3):
    """读取单张图片，返回原图坐标下的边框、得分和类别。"""
    image = read_image(image_path)
    original_size = tuple(image.shape[-2:])

    resized_image = resize_image(image)
    resized_size = tuple(resized_image.shape[-2:])

    # 为单张图片增加批次维度，得到 [1,3,416,416]。
    images = pad_image(resized_image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        predictions = model(images)
        flat = flatten_predictions(predictions)

        # 参考点与预测都按 P3～P7 的顺序合并。
        locations_per_level = make_locations(predictions["cls_logits"], model.strides)
        locations = torch.cat(locations_per_level, dim=0)

        result = postprocess_single_image(
            locations=locations,
            cls_logits=flat["cls_logits"][0],
            box_distances=flat["box_distances"][0],
            center_logits=flat["center_logits"][0],
            resized_size=resized_size,
            original_size=original_size,
            score_threshold=score_threshold,
        )

    return result


def save_prediction_image(image_path, result, class_to_id, save_path):
    """在原图上绘制全部预测框、类别与得分，并保存预览图。"""
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)

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






if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    checkpoint_path = project_dir / "outputs" / "checkpoints" / "best.pth"
    image_path = project_dir / "archive" / "images" / "road0.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model = FCOS(num_classes=checkpoint["num_classes"])
    load_result = model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print("加载的训练轮次：", checkpoint["epoch"])
    print("权重加载结果：", load_result)
    print("预测图片：", image_path.name)
    print("运行设备：", device)

    result = predict_one_image(
        model,
        image_path,
        device,
        score_threshold=0.3,
    )

    print("检测数量：", result["boxes"].shape[0])
    print("前 5 个边框：", result["boxes"][:5])
    print("前 5 个得分：", result["scores"][:5])
    print("前 5 个类别：", result["labels"][:5])
    print(
        "输出形状：",
        result["boxes"].shape,
        result["scores"].shape,
        result["labels"].shape,
    )

    output_path = save_prediction_image(
        image_path=image_path,
        result=result,
        class_to_id=checkpoint["class_to_id"],
        save_path=project_dir / "outputs" / f"{image_path.stem}_pred.png",
    )

    print("预测预览已保存：", output_path)









