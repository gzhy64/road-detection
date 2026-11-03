"""训练 FCOS，记录检测指标，并支持学习率调整和断点续训。"""

import argparse
import csv
import random
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import CLASS_TO_ID, RoadSignDataset, detection_collate_fn
from evaluate import evaluate_model, print_metrics
from experiment import DEFAULT_RUN_DIR, PROJECT_DIR, load_checkpoint, prepare_splits, save_json, select_device
from locations import make_locations
from losses import compute_losses
from model import FCOS
from targets import build_targets


class EpochLRScheduler:
    """按累计完成的轮数衰减学习率，旧检查点也能使用同一计划。"""

    def __init__(self, optimizer, milestones=(30, 40), gamma=0.1, completed_epochs=0):
        self.optimizer = optimizer
        self.milestones = list(milestones)
        self.gamma = gamma
        self.base_lrs = [group.get("initial_lr", group["lr"]) for group in optimizer.param_groups]
        self.last_epoch = completed_epochs
        self._apply()

    def _apply(self):
        factor = self.gamma ** sum(self.last_epoch >= milestone for milestone in self.milestones)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["initial_lr"] = base_lr
            group["lr"] = base_lr * factor

    def step(self):
        self.last_epoch += 1
        self._apply()

    def state_dict(self):
        return {"milestones": self.milestones, "gamma": self.gamma,
                "base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def load_state_dict(self, state):
        for name in ("milestones", "gamma", "base_lrs", "last_epoch"):
            setattr(self, name, state[name])
        self._apply()


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


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, save_path,
                    split_seed, scheduler=None, history=None, best_map50=-1.0,
                    best_epoch=0, best_val_loss=float("inf"), train_generator=None,
                    run_config=None, dataset_metadata=None):
    """在轮次边界保存权重、优化器、学习率计划和可恢复的随机状态。"""
    checkpoint = {
        "format_version": 2, "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses, "val_losses": val_losses,
        "num_classes": model.num_classes, "class_to_id": CLASS_TO_ID.copy(),
        "split_seed": split_seed, "best_map50": best_map50,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "history": history or [], "run_config": run_config or {},
        "dataset_metadata": dataset_metadata,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if train_generator is not None:
        checkpoint["train_generator_state"] = train_generator.get_state()
    if torch.cuda.is_available():
        checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore_random_state(checkpoint, train_generator):
    """新格式检查点恢复随机状态；旧格式仅继续优化，不承诺逐步一致。"""
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "train_generator_state" in checkpoint:
        train_generator.set_state(checkpoint["train_generator_state"])
    if "cuda_rng_states" in checkpoint and torch.cuda.is_available():
        if len(checkpoint["cuda_rng_states"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])


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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=50, help="累计目标轮数，不是追加轮数")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--milestones", type=int, nargs="+")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=0, help="评估无改善的轮数上限，0 表示禁用早停")
    args = parser.parse_args()
    if args.epochs < 1 or args.eval_interval < 1 or args.num_workers < 0 or args.patience < 0:
        parser.error("轮数和评估间隔须为正数，工作进程和早停轮数不能为负数。")
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint and any(getattr(args, key) is not None for key in
                          ("lr", "momentum", "weight_decay", "milestones", "gamma")):
        parser.error("续训会恢复原优化器和学习率计划；修改这些参数请使用新的实验目录从头训练。")
    config = dict(seed=2026, batch_size=4, lr=1e-3, momentum=0.9,
                  weight_decay=1e-4, milestones=[30, 40], gamma=0.1, max_grad_norm=10.0)
    if checkpoint:
        config.update(checkpoint.get("run_config", {}))
        config["seed"] = checkpoint["split_seed"]
        # 老检查点的优化器参数优先于新实验的默认参数。
        group = checkpoint["optimizer_state_dict"]["param_groups"][0]
        for key in ("lr", "momentum", "weight_decay"):
            config[key] = group.get("initial_lr", group["lr"]) if key == "lr" else group.get(key, 0.0)
        if args.seed is not None and args.seed != config["seed"]:
            parser.error("续训时不能更改数据划分种子。")
    for key in ("seed", "batch_size", "lr", "momentum", "weight_decay", "milestones", "gamma", "max_grad_norm"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if (config["batch_size"] < 1 or config["lr"] <= 0 or not 0 <= config["momentum"] < 1
            or config["weight_decay"] < 0 or not 0 < config["gamma"] <= 1
            or any(value < 1 for value in config["milestones"])):
        parser.error("批次大小、学习率或优化器参数不合法。")
    start_epoch = checkpoint["epoch"] if checkpoint else 0
    if start_epoch >= args.epochs:
        parser.error(f"检查点已完成 {start_epoch} 轮，请将 --epochs 设为更大的累计轮数。")
    latest_path = args.output_dir / "latest.pth"
    if latest_path.exists() and (not args.resume or args.resume.resolve() != latest_path.resolve()):
        parser.error("输出目录已有 latest.pth，请从它续训，或选择新的 --output-dir。")
    device = select_device(args.device)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    dataset = RoadSignDataset(args.data_dir)
    indices, dataset_metadata = prepare_splits(dataset, config["seed"], checkpoint)
    if not indices["train"] or not indices["val"]:
        parser.error("训练集和验证集不能为空。")
    train_generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        Subset(dataset, indices["train"]), batch_size=config["batch_size"], shuffle=True,
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
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = EpochLRScheduler(optimizer, config["milestones"], config["gamma"], start_epoch)
    if checkpoint and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint:
        restore_random_state(checkpoint, train_generator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.update(epochs=args.epochs, eval_interval=args.eval_interval, patience=args.patience,
                  device=str(device), num_workers=args.num_workers, data_dir=str(args.data_dir))
    save_json(config, args.output_dir / "config.json")
    save_json(dataset_metadata, args.output_dir / "split.json")
    history = list(checkpoint.get("history", [])) if checkpoint else []
    best_map = checkpoint.get("best_map50", -1.0) if checkpoint else -1.0
    best_epoch = checkpoint.get("best_epoch", 0) if checkpoint else 0
    best_val_loss = checkpoint.get("best_val_loss", float("inf")) if checkpoint else float("inf")
    best_path = args.output_dir / "best.pth"
    if checkpoint and not best_path.exists():
        previous_best = args.resume.parent / "best.pth"
        if previous_best.exists() and checkpoint.get("format_version") == 2:
            shutil.copy2(previous_best, best_path)
        else:
            best_map = -1.0
    print(f"训练设备：{device} | 样本：{len(indices['train'])} | 批次：{len(train_loader)}", flush=True)
    print(f"本次训练第 {start_epoch + 1}～{args.epochs} 轮，输出目录：{args.output_dir}", flush=True)
    if checkpoint and "scheduler_state_dict" not in checkpoint:
        print("旧检查点已加载；按累计轮次启用新学习率计划，旧格式无法精确恢复随机顺序。", flush=True)

    def persist(path, epoch, train_losses, val_losses):
        save_checkpoint(model, optimizer, epoch, train_losses, val_losses, path,
                        config["seed"], scheduler, history, best_map, best_epoch,
                        best_val_loss, train_generator, config, dataset_metadata)

    # 旧格式按验证损失选过 best；首次迁移时先建立检测指标基线。
    if checkpoint and best_map < 0:
        baseline = evaluate_model(model, dataset, indices["val"], device, num_workers=args.num_workers)
        if baseline["map50"] is None:
            raise ValueError("验证集没有真实目标，无法按 mAP50 选择模型。")
        best_map, best_epoch = baseline["map50"], start_epoch
        best_val_loss = checkpoint["val_losses"]["loss_total"]
        save_json(baseline, args.output_dir / f"evaluation_epoch_{start_epoch:04d}.json")
        persist(best_path, start_epoch, checkpoint["train_losses"], checkpoint["val_losses"])
        print(f"续训基线：第 {best_epoch} 轮，验证 mAP50={best_map:.4f}", flush=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
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
        # 轮末调整下一轮学习率，然后连同状态保存，续训不会重复衰减。
        scheduler.step()
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
