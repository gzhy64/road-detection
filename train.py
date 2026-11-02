"""组织 FCOS 的训练过程。"""

from pathlib import Path

import torch

from dataset import CLASS_TO_ID
from locations import make_locations
from losses import compute_losses
from targets import build_targets


def train_one_batch(model, images, targets, optimizer, device):
    """完成一个批次的前向计算、损失计算和参数更新。"""
    model.train()
    images = images.to(device)

    # 只移动张量，图片尺寸等元组信息保持原样。
    for target in targets:
        target["boxes"] = target["boxes"].to(device)
        target["labels"] = target["labels"].to(device)

    optimizer.zero_grad(set_to_none=True)

    predictions = model(images)

    locations = make_locations(predictions["cls_logits"], model.strides)
    batch_targets = build_targets(locations, targets, num_classes=model.num_classes)

    losses = compute_losses(predictions, batch_targets)

    # 遇到无效损失时停止，避免继续更新参数。
    if not torch.isfinite(losses["loss_total"]).item():
        raise RuntimeError("总损失出现 NaN 或 Inf，停止本次更新。")

    losses["loss_total"].backward()
    optimizer.step()

    # 日志只需要数值，将损失张量转换为 Python 数字。
    loss_values = {}
    for name, value in losses.items():
        loss_values[name] = value.item()

    return loss_values


def train_one_epoch(model, train_loader, optimizer, device):
    """遍历训练集一轮，返回各批次损失的平均值。"""
    loss_sums = {
        "loss_cls": 0.0,
        "loss_box": 0.0,
        "loss_center": 0.0,
        "loss_total": 0.0,
    }

    num_batches = len(train_loader)

    for batch_index, (images, targets) in enumerate(train_loader, start=1):
        loss_values = train_one_batch(model, images, targets, optimizer, device)
        # 累加当前批次的四个损失数值。
        for name, value in loss_values.items():
            loss_sums[name] += value

        # 打印第一批、每 50 批以及最后一批。
        # if batch_index == 1 or batch_index % 50 == 0 or batch_index == num_batches:
        #     print(
        #         f"批次 {batch_index}/{num_batches} | "
        #         f"总损失 {loss_values['loss_total']:.4f} | "
        #         f"分类 {loss_values['loss_cls']:.4f} | "
        #         f"回归 {loss_values['loss_box']:.4f} | "
        #         f"中心度 {loss_values['loss_center']:.4f}"
        #     )

    epoch_losses = {}
    for name, value in loss_sums.items():
        epoch_losses[name] = value / num_batches

    return epoch_losses


def validate_one_epoch(model, val_loader, device):
    """遍历验证集，返回各批次损失的平均值。"""
    model.eval()

    loss_sums = {
        "loss_cls": 0.0,
        "loss_box": 0.0,
        "loss_center": 0.0,
        "loss_total": 0.0,
    }

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            for target in targets:
                target["boxes"] = target["boxes"].to(device)
                target["labels"] = target["labels"].to(device)

            predictions = model(images)

            locations = make_locations(predictions["cls_logits"], model.strides)
            batch_targets = build_targets(locations, targets, num_classes=model.num_classes)

            losses = compute_losses(predictions, batch_targets)

            if not torch.isfinite(losses["loss_total"]).item():
                raise RuntimeError("验证总损失出现 NaN 或 Inf")

            for name, value in losses.items():
                loss_sums[name] += value.item()

    epoch_losses = {}
    for name, value in loss_sums.items():
        epoch_losses[name] = value / len(val_loader)

    return epoch_losses


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, save_path, split_seed):
    """保存模型参数、优化器状态和本轮实验信息。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "num_classes": model.num_classes,
        "class_to_id": CLASS_TO_ID.copy(),
        "split_seed": split_seed,
    }

    torch.save(checkpoint, save_path)


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    from dataset import RoadSignDataset, detection_collate_fn, split_dataset
    from model import FCOS

    seed = 2026
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RoadSignDataset(Path(__file__).resolve().parent / "archive")
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, seed=seed)

    # 参数更新使用训练子集
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=detection_collate_fn)

    # 模型和优化器创建一次，在所有批次间持续使用。
    model = FCOS(num_classes=4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    print("训练设备：", device)
    print("训练集样本数：", len(train_dataset))
    print("训练批次数：", len(train_loader))

    num_epochs = 5
    checkpoint_dir = Path(__file__).resolve().parent / "outputs" / "checkpoints"

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n开始第 {epoch}/{num_epochs} 轮训练")

        train_losses = train_one_epoch(model, train_loader, optimizer, device)
        val_losses = validate_one_epoch(model, val_loader, device)

        print(
            f"第 {epoch} 轮结束 | "
            f"训练总损失 {train_losses['loss_total']:.6f} | "
            f"验证总损失 {val_losses['loss_total']:.6f}"
        )

        # 每轮完成后，保存最新训练状态。
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            save_path=checkpoint_dir / "latest.pth",
            split_seed=seed,
        )

        # 验证总损失降低时，保存这一轮作为当前最佳模型。
        current_val_loss = val_losses["loss_total"]

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                save_path=checkpoint_dir / "best.pth",
                split_seed=seed,
            )

            print(
                f"更新 best.pth：第 {best_epoch} 轮，"
                f"验证总损失 {best_val_loss:.6f}"
            )

    print(
        f"\n训练完成，最低验证总损失为 {best_val_loss:.6f}，"
        f"对应第 {best_epoch} 轮。"
    )

    # 读回两个文件，检查实际保存的轮次和损失。
    for file_name in ("latest.pth", "best.pth"):
        checkpoint = torch.load(checkpoint_dir / file_name, map_location="cpu", weights_only=True)

        print(
            f"{file_name} | "
            f"轮次 {checkpoint['epoch']} | "
            f"验证总损失 {checkpoint['val_losses']['loss_total']:.6f}"
        )
