"""管理实验路径、模型权重、数据划分和实验记录。"""

import hashlib
import json
from pathlib import Path

import torch

from dataset import CLASS_TO_ID, split_dataset


PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "outputs" / "runs"
DEFAULT_RUN_DIR = RUNS_DIR / "fcos"


def create_run_directory(epochs, batch_size, lr, augment=False, runs_dir=None):
    """按关键参数创建实验目录，重名时增加序号，保留已有目录中的结果。"""
    root = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    root.mkdir(parents=True, exist_ok=True)
    name = f"fcos_e{epochs}_b{batch_size}_lr{lr:g}"
    if augment:
        name += "_aug"
    number = 1
    while True:
        directory = root / (name if number == 1 else f"{name}_{number}")
        try:
            # 用实际创建操作占用目录名，两个同时启动的实验也不会共用目录。
            directory.mkdir()
        except FileExistsError:
            number += 1
        else:
            return directory


def select_device(name="auto"):
    """默认使用 CUDA，未指定 GPU 或 GPU 不可用时使用 CPU。"""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("指定了 CUDA，但当前 PyTorch 环境无法使用 CUDA。")
    return device


def default_checkpoint():
    """优先使用新实验的最佳模型，兼容之前保存的检查点。"""
    path = DEFAULT_RUN_DIR / "best.pth"
    return path if path.is_file() else PROJECT_DIR / "outputs/checkpoints/best.pth"


def load_checkpoint(path):
    """在 CPU 上读取检查点，并核对类别定义。"""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("class_to_id") != CLASS_TO_ID:
        raise ValueError("检查点的类别映射与 dataset.py 不一致。")
    if checkpoint.get("num_classes") != len(CLASS_TO_ID):
        raise ValueError("检查点的类别数量不正确。")
    return checkpoint


def load_model(path, device):
    """构建网络并加载权重，返回模型及检查点信息。"""
    from model import FCOS

    checkpoint = load_checkpoint(path)
    model = FCOS(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def save_json(data, path):
    """将普通 Python 数据原子写入 JSON，避免留下半个文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_splits(dataset, seed=2026, checkpoint=None):
    """固定划分成员并校验 XML 指纹，返回三个编号列表和划分记录。"""
    paths = dataset.annotation_paths
    if not paths:
        raise ValueError(f"没有找到 XML 标注：{dataset.annotation_dir}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes())
        if not (dataset.image_dir / f"{path.stem}.png").is_file():
            raise FileNotFoundError(f"标注缺少对应图片：{path.name}")

    fingerprint = digest.hexdigest()
    seed = (checkpoint or {}).get("split_seed", seed)
    saved = (checkpoint or {}).get("dataset_metadata")
    if saved:
        if saved["annotation_fingerprint"] != fingerprint:
            raise ValueError("标注内容或文件列表发生变化，不能沿用该检查点的数据划分。")
        split_files = saved["split_files"]
        names_to_indices = {path.name: i for i, path in enumerate(paths)}
        indices = {
            name: [names_to_indices[file] for file in split_files[name]]
            for name in ("train", "val", "test")
        }
        flat = [i for group in indices.values() for i in group]
        if sorted(flat) != list(range(len(dataset))):
            raise ValueError("检查点的数据划分存在重复或遗漏。")
    else:
        subsets = split_dataset(dataset, seed=seed)
        indices = {
            name: list(subset.indices)
            for name, subset in zip(("train", "val", "test"), subsets)
        }

    metadata = {
        "split_seed": seed,
        "annotation_fingerprint": fingerprint,
        "split_files": {
            name: [paths[i].name for i in group] for name, group in indices.items()
        },
    }
    return indices, metadata
