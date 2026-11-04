# 路标目标检测：手写 FCOS

使用 PyTorch 基础层实现四类路标的类别识别与边框定位。主干网络、FPN、检测头、目标分配、损失、解码和 NMS 均在本项目中实现，没有调用现成检测器，也没有使用预训练权重。

这是 **FCOS 思路的教学变体**：使用自定义 TinyBackbone，保留五级特征金字塔、逐位置 ltrb 回归、中心度和 Focal Loss。它并非原论文 ResNet-50 模型的严格复现，也不承诺论文中的精度。

首次运行的顺序是：**准备环境和数据 → 从头训练 → 验证集评估与可视化 → 最终测试集评估**。初始没有 `.pth` 权重文件时，直接训练即可；后续命令统一使用训练生成的 `outputs/runs/fcos/best.pth`。

## 1. 准备环境和数据

正式训练环境是服务器 Python 3.10、PyTorch 2.7.1+cu118、Tesla V100 16GB。使用服务器已有的训练环境；`requirements.txt` 记录依赖，绘制训练曲线需要 matplotlib。

在服务器项目目录更新代码，后面的命令都在该目录、已激活的训练环境中执行：

```bash
cd /home/mel/zhy/road-detection
git pull --ff-only origin main
```

将 `archive.zip` 解压为：

```text
road-detection/
├── archive/
│   ├── images/          # road0.png …
│   └── annotations/     # road0.xml …，VOC 格式
├── train.py
├── evaluate.py
└── predict.py
```

类别编号为 `trafficlight=0, speedlimit=1, crosswalk=2, stop=3`。877 张图片按种子 2026 划分为训练 701、验证 87、测试 89 张。

- 训练集用于更新网络参数。
- 验证集用于选择模型和调整参数。
- 测试集用于模型及推理参数确定后的最终评价。

模型权重保存具体划分文件名、划分种子及 XML 指纹，评估时沿用对应划分。数据集、权重和实验输出均被 Git 忽略，首次获取项目代码后需要自行准备数据并训练。

## 2. 从头训练，生成模型权重

```bash
python3 train.py --epochs 50 --batch-size 4 --seed 2026 --output-dir outputs/runs/fcos
```

每次运行都会创建随机初始化的模型，从第 1 轮开始训练。默认 SGD：初始学习率 0.001、momentum 0.9、weight decay 0.0001、梯度范数上限 10。

学习率只使用 `CosineAnnealingLR` 余弦退火，每轮结束后平滑下调下一轮的学习率，无周期重启。`--lr` 设置初始值，`--min-lr` 设置最低值（默认 `1e-6`），须满足 `0 < min_lr <= lr`。训练至少两轮且正常完成时，第 1 轮使用初始值，最后一轮使用最低值；只训练一轮时使用初始值，提前停止时可能尚未降到最低值。原来的 `--milestones`、`--gamma` 已删除。

例如，进行 100 轮余弦退火实验：

```bash
python3 train.py --epochs 100 --batch-size 4 --seed 2026 --lr 0.001 --min-lr 1e-6 --output-dir outputs/runs/fcos100_cosine
```

该命令的模型权重、评估和绘图路径应使用 `outputs/runs/fcos100_cosine/`。初始值、最低值及调度方式会写入 `config.json` 和模型权重中的 `run_config`；`history.json/csv` 中的 `lr` 是每轮实际用于参数更新的值。是否优于既有实验，以相同数据划分下的验证集指标判断。

训练过程中，每轮计算训练和验证损失，在第 1 轮、每 5 轮及最后一轮计算验证集检测指标，用来选择最佳权重。可用 `--eval-interval 1` 每轮评估；`--patience 15` 可启用评估无改善后的早停，默认禁用。测试集不参与这一过程。

结果写入 `outputs/runs/fcos/`：

| 文件 | 用途 |
|---|---|
| `best.pth` | 已评估轮次中验证 mAP50 最高的模型；同分时选择验证损失更低者 |
| `latest.pth` | 最新一轮的模型权重 |
| `config.json`、`split.json` | 本次训练配置和数据划分 |
| `history.json`、`history.csv` | 训练损失、验证损失及检测指标记录 |
| `evaluation_epoch_*.json` | 各次验证集检测评估结果 |

权重文件供预测和评估使用。输出目录已有训练结果时，请通过 `--output-dir` 指定新目录，例如 `outputs/runs/fcos_new`；下面各命令中的权重和历史路径也应相应替换。

## 3. 验证集评估与预测可视化

### 汇总验证集指标

训练完成、生成 `best.pth` 后，执行：

```bash
python3 evaluate.py --checkpoint outputs/runs/fcos/best.pth --split val --output outputs/runs/fcos/val_metrics.json
```

该命令对权重对应划分中的 87 张验证图片进行评估，打印四个类别的 AP50、精确率、召回率、TP/FP/FN，并保存 JSON。训练过程中已进行的验证用于选择权重，这里单独汇总所选最佳模型的结果。

查看训练和验证曲线：

```bash
python3 plot_history.py --history outputs/runs/fcos/history.json --output outputs/runs/fcos/training_curves.png
```

根据验证集结果选择模型和推理阈值；如果调整了设置并重新训练，重复本步骤进行比较。

### 绘制验证集图片

下面的图片名对应原始数据集、划分种子 2026 下的验证集样本。更换数据集或种子后，请根据本次实验的 `split.json` 中 `split_files.val` 选择图片。

绘制一张验证集图片：

```bash
python3 predict.py --checkpoint outputs/runs/fcos/best.pth --image archive/images/road261.png --annotation archive/annotations/road261.xml --output outputs/validation_predictions/road261_compare.png
```

一次绘制多张验证集图片：

```bash
python3 predict.py --checkpoint outputs/runs/fcos/best.pth --image archive/images/road261.png archive/images/road141.png archive/images/road415.png --annotation-dir archive/annotations --output-dir outputs/validation_predictions --batch-size 2
```

预览中预测框为红色、真实框为绿色；每张图旁的同名 JSON 保存完整浮点边框、类别和得分。XML 只在预测完成后用于绘制绿色真实框。省略 `--annotation` 或 `--annotation-dir` 即只画预测框。

模型只加载一次，按 `--batch-size` 分批处理。多图输出使用 `--output-dir`；指定标注文件夹时，每张图片需要有同名 XML。绘图用于观察误检、漏检和定位偏差，完整验证集指标用于比较模型。

文件夹绘图也支持 `--image-dir`、`--limit` 和 `--batch-size`。该模式只按文件名收集图片，不会自动按数据子集筛选；如果需要验证集展示，应指向单独整理的验证集图片目录。直接读取 `archive/images` 会包含训练、验证和测试图片，不能将整组预览称为验证集或测试集结果。

## 4. 最终测试集评估

在训练配置、最佳权重和推理阈值确定后，最后评估测试集：

```bash
python3 evaluate.py --checkpoint outputs/runs/fcos/best.pth --split test --output outputs/runs/fcos/test_metrics.json
```

该命令只使用权重对应划分中的 89 张测试图片。最终报告使用测试集指标，并注明模型权重和推理参数；训练集示例、验证集成绩和测试集成绩分别标注。

如果验证阶段调整了 `--score-threshold`、`--report-threshold` 或 `--nms-iou-threshold`，最终测试时使用同样的值。测试集用于最终评价，参数选择以验证集为依据。

## 指标与输出说明

指标定义：

- **AP50**：按类别，在 IoU ≥ 0.5 下将预测按得分排序，匹配同一图片中 IoU 最大的同类别真实框。真实框已匹配时，后续预测计为 FP。采用 VOC 2010+ 的全点精确率包络积分。
- **mAP50**：有真实目标的类别 AP50 的算术平均，不等于 COCO mAP@[.5:.95]。缺少真实目标的类别 AP 为 `null`，不参与平均。
- **精确率、召回率、F1、TP/FP/FN**：默认在预测得分 ≥ 0.3 时统计。总体 P/R 对各类别计数求和后计算。
- AP 的候选得分阈值默认 0.05，P/R 得分阈值默认 0.3，可用 `--score-threshold`、`--report-threshold` 调整。候选数量上限和阈值会影响结果，报告中保存完整评估配置。
- 原图坐标采用连续 `xyxy`，面积不加 1；因此不是 VOC 官方整数坐标评估器的逐位复现。
- 按类别 NMS 的 IoU 阈值默认 0.5，每图 NMS 前最多 1000 个候选，最终最多 100 个检测框。不同类别不相互抑制；每张图允许检测多个目标。

实验目录包含 `config.json`、`split.json`、`history.json/csv`、逐次评估 JSON，以及 `best.pth/latest.pth`。损失日志是批次平均值，并非将整轮所有正样本统一归一化后重新计算的损失。

## 模块与学习资料

输入统一为 `[B,3,416,416]`，模型输出五个尺度：

| 特征 | 步长 | 主干通道数 | FPN 输出 | 类别/距离/中心度通道 |
|---|---:|---:|---|---|
| C3 → P3 | 8 | 128 | `[B,256,52,52]` | 4 / 4 / 1 |
| C4 → P4 | 16 | 256 | `[B,256,26,26]` | 4 / 4 / 1 |
| C5 → P5 | 32 | 512 | `[B,256,13,13]` | 4 / 4 / 1 |
| P6 | 64 | — | `[B,256,7,7]` | 4 / 4 / 1 |
| P7 | 128 | — | `[B,256,4,4]` | 4 / 4 / 1 |

五级共有 3614 个参考点，总参数量 12,642,222。函数之间的数据流、坐标和标签约定见 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)。训练、预测、评估入口直接操作真实数据。

参考：[FCOS 原论文](https://arxiv.org/abs/1904.01355)、[FCOS 作者实现](https://github.com/tianzhi0549/FCOS)、[VOC AP 积分代码](https://github.com/weiliu89/VOCdevkit/blob/master/VOCcode/VOCap.m)。
