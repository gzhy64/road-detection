"""从训练历史绘制损失、验证集检测指标和逐类别 AP50 曲线。"""

import argparse
import json
from pathlib import Path

from experiment import DEFAULT_RUN_DIR


def plot_history(history_path, output_path):
    """读取 history.json，跳过未执行检测评估的轮次。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    if not history:
        raise ValueError("训练历史为空。")
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss_total"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss_total"] for row in history], label="Validation")
    axes[0].set(title="Loss", ylabel="Batch mean loss")
    evaluated = [row for row in history if row.get("map50") is not None]
    evaluated_epochs = [row["epoch"] for row in evaluated]
    for key, label in (("map50", "mAP50"), ("precision", "Precision"), ("recall", "Recall")):
        axes[1].plot(evaluated_epochs, [row[key] for row in evaluated], marker=".", label=label)
    axes[1].set(title="Validation detection metrics", ylabel="Score", ylim=(0, 1))
    for name in ("trafficlight", "speedlimit", "crosswalk", "stop"):
        rows = [row for row in evaluated if row.get(f"ap50_{name}") is not None]
        axes[2].plot([row["epoch"] for row in rows], [row[f"ap50_{name}"] for row in rows],
                     marker=".", label=name)
    axes[2].set(title="Validation AP50 by class", ylabel="AP50", ylim=(0, 1))
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.legend()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_RUN_DIR / "history.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN_DIR / "training_curves.png")
    args = parser.parse_args()
    print("训练曲线已保存：", plot_history(args.history, args.output))


if __name__ == "__main__":
    main()
