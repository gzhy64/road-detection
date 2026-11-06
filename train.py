"""从头训练 FCOS，记录检测指标，并保存用于预测的模型权重。"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from augmentations import TrainAugmentation
from dataset import CLASS_TO_ID, RoadSignDataset, detection_collate_fn
from evaluate import evaluate_model, print_metrics
from experiment import PROJECT_DIR, create_run_directory, prepare_splits, save_json, select_device
from locations import make_locations
from losses import compute_losses
from model import FCOS
from targets import build_targets


def move_targets(targets, device):
    """保留尺寸元数据，只移动训练所需的边框和类别张量。"""
    return [dict(target, boxes=target["boxes"].to(device), labels=target["labels"].to(device))
            for target in targets]


def train_one_batch(model, images, targets, optimizer, device, max_grad_norm=10.0):
    """完成一个批次的损失计算和参数更新，并检查损失与梯度。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(images.to(device))
    locations = make_locations(predictions["cls_logits"], model.strides)
    batch_targets = build_targets(locations, move_targets(targets, device), model.num_classes)
    losses = compute_losses(predictions, batch_targets)
    if not all(torch.isfinite(value).item() for value in losses.values()):
        raise RuntimeError("训练损失出现 NaN 或 Inf，未执行本批次参数更新。")
    losses["loss_total"].backward()
    # 即使关闭裁剪，也检查梯度，避免将无效梯度写入权重。
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm if max_grad_norm > 0 else float("inf"),
        error_if_nonfinite=True,
    )
    optimizer.step()
    return {name: value.item() for name, value in losses.items()}


def train_one_epoch(model, train_loader, optimizer, device, max_grad_norm=10.0):
    """遍历训练集，返回四项损失的批次平均值。"""
    if len(train_loader) == 0:
        raise ValueError("训练数据加载器为空。")
    sums = dict.fromkeys(("loss_cls", "loss_box", "loss_center", "loss_total"), 0.0)
    for images, targets in train_loader:
        values = train_one_batch(model, images, targets, optimizer, device, max_grad_norm)
        for name, value in values.items():
            sums[name] += value
    return {name: value / len(train_loader) for name, value in sums.items()}


@torch.no_grad()
def validate_one_epoch(model, val_loader, device):
    """计算验证损失；检测效果另外用 AP50、精确率和召回率衡量。"""
    if len(val_loader) == 0:
        raise ValueError("验证数据加载器为空。")
    model.eval()
    sums = dict.fromkeys(("loss_cls", "loss_box", "loss_center", "loss_total"), 0.0)
    for images, targets in val_loader:
        predictions = model(images.to(device))
        locations = make_locations(predictions["cls_logits"], model.strides)
        batch_targets = build_targets(locations, move_targets(targets, device), model.num_classes)
        losses = compute_losses(predictions, batch_targets)
        if not all(torch.isfinite(value).item() for value in losses.values()):
            raise RuntimeError("验证损失出现 NaN 或 Inf。")
        for name, value in losses.items():
            sums[name] += value.item()
    return {name: value / len(val_loader) for name, value in sums.items()}


def save_model(model, epoch, train_losses, val_losses, save_path,
               split_seed, run_config, dataset_metadata):
    """保存模型参数和评估所需信息，供后续预测、评估使用。"""
    weights = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "num_classes": model.num_classes,
        "class_to_id": CLASS_TO_ID.copy(),
        "split_seed": split_seed,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "run_config": run_config,
        "dataset_metadata": dataset_metadata,
    }
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(weights, temporary)
    temporary.replace(path)


def write_history(history, directory):
    """同时保存完整 JSON 历史和便于画图、整理实验表的 CSV。"""
    save_json(history, directory / "history.json")
    fields = list(dict.fromkeys(key for row in history for key in row))
    temporary = directory / "history.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(directory / "history.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "archive")
    parser.add_argument("--output-dir", type=Path, help="自定义保存目录，默认按轮数、批次和初始学习率自动命名")
    parser.add_argument("--epochs", type=int, default=50, help="本次从头训练的轮数")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float, help="初始学习率，默认 0.001")
    parser.add_argument("--min-lr", type=float, help="学习率下限，默认 0.000001，不要求训练结束时达到")
    parser.add_argument("--lr-decay", type=float, help="每轮学习率的乘数，默认 0.95，须大于 0 且小于 1")
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=0, help="评估无改善的轮数上限，0 表示禁用早停")
    args = parser.parse_args()
    if args.epochs < 1 or args.eval_interval < 1 or args.num_workers < 0 or args.patience < 0:
        parser.error("轮数和评估间隔须为正数，工作进程和早停轮数不能为负数。")
    config = dict(seed=2026, batch_size=4, lr=1e-3, min_lr=1e-6, lr_decay=0.95, momentum=0.9,
                  weight_decay=1e-4, max_grad_norm=10.0)
    for key in ("seed", "batch_size", "lr", "min_lr", "lr_decay", "momentum", "weight_decay", "max_grad_norm"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if (config["batch_size"] < 1 or not math.isfinite(config["lr"]) or config["lr"] <= 0
            or not 0 <= config["momentum"] < 1 or config["weight_decay"] < 0):
        parser.error("批次大小、学习率或优化器参数不合法。")
    if not 0 < config["min_lr"] <= config["lr"]:
        parser.error("最低学习率须为有限正数，且不能大于初始学习率。")
    if not 0 < config["lr_decay"] < 1:
        parser.error("学习率每轮衰减乘数须大于 0 且小于 1。")
    if args.output_dir is not None and any(
        (args.output_dir / name).exists() for name in ("latest.pth", "best.pth", "config.json")
    ):
        parser.error("输出目录已有训练结果，请通过 --output-dir 指定新的实验目录。")
    device = select_device(args.device)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    dataset = RoadSignDataset(args.data_dir)
    indices, dataset_metadata = prepare_splits(dataset, config["seed"])
    if not indices["train"] or not indices["val"]:
        parser.error("训练集和验证集不能为空。")
    augmentation = TrainAugmentation()
    # 增强使用独立的数据集实例，验证损失和检测评估始终读取基础实例。
    train_dataset = RoadSignDataset(args.data_dir, transform=augmentation)
    train_generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        Subset(train_dataset, indices["train"]), batch_size=config["batch_size"], shuffle=True,
        num_workers=args.num_workers, collate_fn=detection_collate_fn, generator=train_generator,
    )
    val_loader = DataLoader(
        Subset(dataset, indices["val"]), batch_size=2, shuffle=False,
        num_workers=args.num_workers, collate_fn=detection_collate_fn,
        generator=torch.Generator().manual_seed(0),
    )
    model = FCOS(num_classes=len(CLASS_TO_ID)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    if args.output_dir is None:
        args.output_dir = create_run_directory(
            args.epochs, config["batch_size"], config["lr"], augment=True,
        )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "latest.pth"
    config.update(scheduler="exponential", epochs=args.epochs,
                  eval_interval=args.eval_interval, patience=args.patience,
                  device=str(device), num_workers=args.num_workers, data_dir=str(args.data_dir),
                  output_dir=str(args.output_dir.resolve()))
    config["augmentation"] = augmentation.to_config()
    save_json(config, args.output_dir / "config.json")
    save_json(dataset_metadata, args.output_dir / "split.json")
    history = []
    best_map = -1.0
    best_epoch = 0
    best_val_loss = float("inf")
    best_path = args.output_dir / "best.pth"
    print(f"训练设备：{device} | 样本：{len(indices['train'])} | 批次：{len(train_loader)}", flush=True)
    print(f"从第 1 轮开始训练，共 {args.epochs} 轮，输出目录：{args.output_dir}", flush=True)
    print(f"学习率：初始值 {config['lr']:.6g}，每轮乘 {config['lr_decay']:.6g}，"
          f"下限 {config['min_lr']:.6g}", flush=True)
    print("训练增强：默认启用", flush=True)

    def persist(path, epoch, train_losses, val_losses):
        save_model(model, epoch, train_losses, val_losses, path,
                   config["seed"], config, dataset_metadata)

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_losses = train_one_epoch(model, train_loader, optimizer, device, config["max_grad_norm"])
        val_losses = validate_one_epoch(model, val_loader, device)
        row = {"epoch": epoch, "lr": learning_rate,
               **{f"train_{key}": value for key, value in train_losses.items()},
               **{f"val_{key}": value for key, value in val_losses.items()}}
        improved = False
        evaluated = epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs
        if evaluated:
            metrics = evaluate_model(model, dataset, indices["val"], device, num_workers=args.num_workers)
            if metrics["map50"] is None:
                raise ValueError("验证集没有真实目标，无法按 mAP50 选择模型。")
            row.update(map50=metrics["map50"], precision=metrics["precision"], recall=metrics["recall"])
            row.update({f"ap50_{name}": value["ap50"] for name, value in metrics["classes"].items()})
            save_json(metrics, args.output_dir / f"evaluation_epoch_{epoch:04d}.json")
            improved = metrics["map50"] > best_map or (
                metrics["map50"] == best_map and val_losses["loss_total"] < best_val_loss)
            if improved:
                best_map, best_epoch = metrics["map50"], epoch
                best_val_loss = val_losses["loss_total"]
            print_metrics(metrics)
        row["seconds"] = time.perf_counter() - started
        history.append(row)
        # 每轮按固定比例降低下一轮的学习率；最低值只作下限，不作训练终点。
        if epoch < args.epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(config["min_lr"], param_group["lr"] * config["lr_decay"])
        persist(latest_path, epoch, train_losses, val_losses)
        if improved:
            persist(best_path, epoch, train_losses, val_losses)
        write_history(history, args.output_dir)
        print(f"第 {epoch}/{args.epochs} 轮 | lr={learning_rate:.6g} | "
              f"训练损失 {train_losses['loss_total']:.4f} | 验证损失 {val_losses['loss_total']:.4f} | "
              f"耗时 {row['seconds']:.1f}s", flush=True)
        if improved:
            print(f"更新 best.pth：第 {best_epoch} 轮，验证 mAP50={best_map:.4f}", flush=True)
        if evaluated and args.patience and epoch - best_epoch >= args.patience:
            print(f"检测指标已连续 {epoch - best_epoch} 轮未改善，提前停止。", flush=True)
            break
    print(f"训练完成。最佳轮次：{best_epoch}，验证 mAP50：{best_map:.4f}", flush=True)


if __name__ == "__main__":
    main()
