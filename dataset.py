"""
读取路标图片和标注，并转换为训练使用的张量
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

CLASS_TO_ID = {
    "trafficlight": 0,
    "speedlimit": 1,
    "crosswalk": 2,
    "stop": 3,
}


def read_annotation(annotation_path):
    """读取一份 XML 标注，返回目标边框和类别张量"""
    root = ET.parse(annotation_path).getroot()

    boxes = []
    labels = []

    for obj in root.iter("object"):
        class_name = obj.findtext("name")
        class_id = CLASS_TO_ID[class_name]
        labels.append(class_id)

        box = obj.find("bndbox")
        coordinates = [
            float(box.findtext("xmin")),
            float(box.findtext("ymin")),
            float(box.findtext("xmax")),
            float(box.findtext("ymax")),
        ]
        boxes.append(coordinates)

    boxes = torch.tensor(boxes, dtype=torch.float32, device="cpu")
    boxes = boxes.reshape(-1, 4)
    labels = torch.tensor(labels, dtype=torch.int64, device="cpu")

    return {
        "boxes": boxes,
        "labels": labels,
    }


def read_image(image_path):
    """读取图片，返回像素范围为 [0, 1] 的 [C, H, W] 浮点张量。"""
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")

    image_array = np.array(image_rgb)
    image_tensor = torch.from_numpy(image_array)
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.float() / 255.0

    return image_tensor


def resize_image(image_tensor, image_size=416):
    """按比例缩放图片，返回 [C, H, W] 张量。"""
    height, width = image_tensor.shape[-2:]
    scale = image_size / max(height, width)

    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    new_size = (new_height, new_width)

    resized = F.interpolate(
        image_tensor.unsqueeze(0),
        size=new_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    return resized.squeeze(0)


def resize_boxes(boxes, old_size, new_size):
    """根据缩放前后的（高，宽）调整边框坐标。"""
    old_height, old_width = old_size
    new_height, new_width = new_size

    scale_x = new_width / old_width
    scale_y = new_height / old_height

    scales = torch.tensor(
        [scale_x, scale_y, scale_x, scale_y],
        dtype=boxes.dtype,
        device=boxes.device,
    )

    return boxes * scales


def pad_image(image_tensor, image_size=416):
    """将已缩放的图片在右侧和下方补零至指定大小。"""
    height, width = image_tensor.shape[-2:]
    pad_right = image_size - width
    pad_bottom = image_size - height

    return F.pad(
        image_tensor,
        (0, pad_right, 0, pad_bottom),
        mode="constant",
        value=0.0,
    )


def prepare_sample(image_tensor, target, image_size=416):
    """同步缩放图片与边框，再将图片补齐到指定大小。"""
    old_size = image_tensor.shape[-2:]
    resized_image = resize_image(image_tensor, image_size)

    new_target = target.copy()

    # 边框比例使用缩放后、补边前的图片尺寸。
    new_target["boxes"] = resize_boxes(
        target["boxes"],
        old_size,
        resized_image.shape[-2:],
    )

    padded_image = pad_image(resized_image, image_size)

    return padded_image, new_target


class RoadSignDataset(Dataset):
    """按编号读取并预处理图片及目标标注。"""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.annotation_dir = self.data_dir / "annotations"
        # 固定样本顺序，后续通过编号找到对应的标注。
        self.annotation_paths = sorted(self.annotation_dir.glob("*.xml"))

    def __len__(self):
        return len(self.annotation_paths)

    def __getitem__(self, idx):
        """按样本编号返回图片张量和目标标注。"""
        annotation_path = self.annotation_paths[idx]
        image_path = self.image_dir / f"{annotation_path.stem}.png"

        image_tensor = read_image(image_path)
        target = read_annotation(annotation_path)

        image_tensor, target = prepare_sample(image_tensor, target)

        return image_tensor, target


def detection_collate_fn(batch):
    """将检测样本整理为图片批次张量和标注列表。"""
    images = []
    targets = []

    for image_tensor, target in batch:
        images.append(image_tensor)
        targets.append(target)

    images = torch.stack(images, dim=0)

    return images, targets


def split_dataset(dataset, seed=42):
    """按约 8:1:1 划分数据，返回训练、验证和测试子集。"""
    split_generator = torch.Generator()
    split_generator.manual_seed(seed)

    indices = torch.randperm(len(dataset), generator=split_generator).tolist()

    num_samples = len(dataset)
    train_size = int(num_samples * 0.8)
    val_size = int(num_samples * 0.1)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    dataset = RoadSignDataset(project_dir / "archive")
    # 划分数据集
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    #训练,验证，测试loader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=detection_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=detection_collate_fn)
    print("验证集批次数量：", len(val_loader))
    print("测试集批次数量：", len(test_loader))



