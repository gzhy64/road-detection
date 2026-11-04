# 路标目标检测：手写 FCOS

使用 PyTorch 基础层实现四类路标的类别识别与边框定位。主干网络、FPN、检测头、目标分配、损失、解码和 NMS 均在本项目中实现，没有调用现成检测器，也没有使用预训练权重。

这是 **FCOS 思路的教学变体**：使用自定义 TinyBackbone，保留五级特征金字塔、逐位置 ltrb 回归、中心度和 Focal Loss。它并非原论文 ResNet-50 模型的严格复现，也不承诺论文中的精度。

## 环境和数据

正式训练环境是服务器 Python 3.10、PyTorch 2.7.1+cu118、Tesla V100 16GB。继续使用服务器已有的 PyTorch 环境；`requirements.txt` 记录依赖。绘图额外需要 matplotlib。

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

类别编号为 `trafficlight=0, speedlimit=1, crosswalk=2, stop=3`。877 张图片按种子 2026 划分为训练 701、验证 87、测试 89 张。模型权重文件保存具体划分文件名及 XML 指纹，供评估时校验。已有权重按文件中的 `split_seed` 确定评估划分，因此需保留相同数据集。

数据集、权重、图像和实验输出均忽略上传，仅提交代码与文档。

## 服务器运行

先在服务器项目目录更新代码，以下命令均使用已经激活的训练环境中的 Python：

```bash
git pull --ff-only origin main
```

先用已有权重检查真实图片及完整验证集，命令中的路径明确指向此前的检查点：

```bash
python3 predict.py --checkpoint outputs/checkpoints/best.pth --image archive/images/road0.png --annotation archive/annotations/road0.xml --output outputs/road0_compare.png
python3 evaluate.py --checkpoint outputs/checkpoints/best.pth --image archive/images/road0.png
python3 evaluate.py --checkpoint outputs/checkpoints/best.pth --split val
```

预览中预测框为红色，真实框为绿色；同名 JSON 保存完整浮点边框、类别和得分。`--annotation` 仅影响绘图，预测本身不需要标注。单图检查不能替代验证集指标。

### 开始训练

```bash
python3 train.py --epochs 50 --batch-size 4 --output-dir outputs/runs/fcos
```

每次运行都创建随机初始化的模型，从第 1 轮开始训练，默认共 50 轮。

默认 SGD：学习率 0.001、momentum 0.9、weight decay 0.0001、梯度范数上限 10。使用 PyTorch 的 `MultiStepLR`，第 30、40 轮完成后学习率各乘 0.1；第 31 轮开始使用 0.0001，第 41 轮开始使用 0.00001。实际参数记录在 `config.json`。

每轮计算训练和验证损失，在第 1 轮、每 5 轮及最后一轮计算验证检测指标。可用 `--eval-interval 1` 每轮评估；`--patience 15` 可启用评估无改善后的早停，默认禁用。

`best.pth` 在已评估轮次中按 mAP50 选择，mAP50 相同时选择验证损失更低者；`latest.pth` 保存最新一轮模型权重。这两个文件供预测和评估使用，只保存网络参数与必要实验信息。

输出目录已有训练结果时，请通过 `--output-dir` 指定新的目录，例如 `outputs/runs/fcos_new`。

## 评估和曲线

```bash
python3 evaluate.py --checkpoint outputs/runs/fcos/best.pth --split val
python3 plot_history.py --history outputs/runs/fcos/history.json --output outputs/runs/fcos/training_curves.png
```

验证集用于选择模型及阈值。模型和推理参数确定后，再对测试集进行最终评估：

```bash
python3 evaluate.py --checkpoint outputs/runs/fcos/best.pth --split test
```

指标定义：

- **AP50**：按类别，在 IoU ≥ 0.5 下将预测按得分排序，匹配同一图片中 IoU 最大的同类别真实框。真实框已匹配时，后续预测计为 FP。采用 VOC 2010+ 的全点精确率包络积分。
- **mAP50**：有真实目标的类别 AP50 的算术平均，不等于 COCO mAP@[.5:.95]。缺少真实目标的类别 AP 为 `null`，不参与平均。
- **精确率、召回率、F1、TP/FP/FN**：默认在预测得分 ≥ 0.3 时统计。总体 P/R 对各类别计数求和后计算。
- AP 的候选得分阈值默认 0.05，P/R 得分阈值默认 0.3，可用 `--score-threshold`、`--report-threshold` 调整。候选数量上限和阈值会影响结果，报告中保存完整评估配置。
- 原图坐标采用连续 `xyxy`，面积不加 1；因此不是 VOC 官方整数坐标评估器的逐位复现。
- 按类别 NMS 的 IoU 阈值默认 0.5，每图 NMS 前最多 1000 个候选，最终最多 100 个检测框。不同类别不相互抑制；每张图允许检测多个目标。

实验目录包含 `config.json`、`split.json`、`history.json/csv`、逐次评估 JSON，以及 `best.pth/latest.pth`。损失日志是批次平均值，并非将整轮所有正样本统一归一化后重新计算的损失。

未指定检查点时，预测和评估优先查找 `outputs/runs/fcos/best.pth`，不存在时兼容旧的 `outputs/checkpoints/best.pth`。比较实验时建议始终显式传入 `--checkpoint`。

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
