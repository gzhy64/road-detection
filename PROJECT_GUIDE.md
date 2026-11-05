# 项目代码与数据流

## 数据进入网络之前

`dataset.py` 中，`RoadSignDataset.__getitem__()` 根据编号寻找同名图片和 XML。它调用 `read_image()` 得到 CPU 上的 `float32 [3,H,W]`、像素范围 `[0,1]`，调用 `read_annotation()` 得到 `boxes: float32 [N,4]` 和 `labels: int64 [N]`。

两者交给 `prepare_sample()`：

1. `resize_image()` 将长边缩放到 416，并对实际高宽取整。
2. `resize_boxes()` 使用实际缩放前后的高宽，分别缩放 x/y 坐标。
3. `pad_image()` 在右边、下边补零至 416×416。
4. 保留 `(H,W)` 顺序的 `original_size` 和补边前的 `resized_size`。

`detection_collate_fn()` 将多张图片堆为 `[B,3,416,416]`，标注仍是长度 B 的字典列表，因为不同图片的目标数不同。DataLoader 负责选择哪些样本属于一个批次，collate 函数负责如何将这些样本整理在一起。

## 网络如何生成预测

`model.FCOS.forward()` 依次调用三个模块：

```text
图片 → backbone.TinyBackbone → C3/C4/C5
     → fpn.FeaturePyramid → P3/P4/P5/P6/P7
     → head.FCOSHead → 分类 logits、正数 ltrb 距离、中心度 logits
```

主干中的 `ConvBlock` 是卷积、GroupNorm 和 ReLU。每个阶段先以 stride=2 下采样，再用 stride=1 提取特征。C3/C4/C5 的通道为 128/256/512。

FPN 通过 1×1 卷积把通道统一成 256，将深层特征用最近邻插值放大，与相邻浅层特征逐元素相加。输出的 3×3 卷积分别学习各尺度的局部整合。P6/P7 继续下采样得到。

检测头的分类塔和回归塔各有四个卷积块；五个尺度共用塔和输出卷积的权重。分类输出 4 通道；回归输出 4 通道，并通过每尺度可学习系数与 `exp()` 转成正距离；中心度输出 1 通道，连接在分类塔上。

分类和中心度的 logits 尚未经过 sigmoid。初始分类偏置对应概率 0.01，用于减少训练起点的大量背景位置对分类损失的影响。ltrb 直接使用输入图像像素单位，解码时不再乘 stride。

## 参考点和训练目标

`locations.make_locations()` 根据各尺度尺寸和 stride 生成 `[Mi,2]` 的 `(x,y)` 参考点。第 row 行、column 列对应：

```text
x = (column + 0.5) × stride
y = (row + 0.5) × stride
```

这是输入图像坐标系中约定的网格参考点，不是物体中心，也不是精确的感受野中心。P6/P7 的某些参考点可能位于输入范围外，必须用有效区域掩码忽略。

`targets.build_targets()` 遍历尺度和图片，为参考点寻找真实框：

- `compute_box_distances()` 计算点到每个真实框的 `[l,t,r,b]`，返回 `[Mi,N,4]`。
- `get_candidate_mask()` 要求点严格位于框内，且最大方向距离属于本尺度回归范围。
- `get_valid_location_mask()` 排除补边和图外参考点。
- `match_boxes_by_area()` 在多个合格框中选择面积最小者。
- `compute_centerness()` 根据四个真实距离计算中心度。

训练标签中 `0～3` 是四个前景类别，`4` 表示背景，`-1` 表示忽略。背景参与分类损失；回归和中心度仅使用正样本。

每尺度输出 `labels [B,Mi]`、`box_distances [B,Mi,4]`、`centerness [B,Mi]`。P3～P7 的 Mi 分别为 2704、676、169、49、16。

## 损失和一次参数更新

`losses.flatten_predictions()` 将网络的 `[B,C,H,W]` 按位置重排，再按 P3～P7 合并为分类 `[B,3614,4]`、距离 `[B,3614,4]`、中心度 `[B,3614]`。参考点与目标必须使用同一尺度及行列顺序。

`compute_losses()` 计算：

- `sigmoid_focal_loss()`：独立类别的二元分类，忽略 -1，按正样本数量归一化。
- `iou_loss()`：预测 ltrb 与真实 ltrb 使用同一个参考点，计算 `-log(IoU)`；以真实中心度加权，并除以权重和。
- `centerness_loss()`：正样本上的 BCEWithLogits，按正样本数量归一化。

没有正样本时，回归和中心度损失为可反向传播的 0；分类仍能学习背景。`train_one_batch()` 负责清梯度、前向、构造目标、损失、反向、梯度检查及裁剪、优化器更新。

## 如何得到最终检测框

`predict.predict_batch()` 进行前向预测、重排和坐标生成，再为每张图片调用 `postprocess_single_image()`：

```text
有效区域筛选 → decode_boxes → filter_candidates
             → clip_candidates_to_image → 候选数量限制
             → nms_by_class → restore_candidates_to_original
```

`decode_boxes()` 根据参考点 `(x,y)` 生成 `[x-l,y-t,x+r,y+b]`。`filter_candidates()` 用 `sqrt(sigmoid(cls) × sigmoid(center))` 排序、筛选；一个点可以有多个类别过阈值，因此可能返回相同坐标、不同类别的候选。

`clip_candidates_to_image()` 裁剪到补边前的有效区域，并删除零面积框。`nms_by_class()` 分类别调用 `nms_single_class()`，删除与已保留高分框 IoU 过大的低分框。不同类别之间不删除，同类别但 IoU 较低的局部框也可能保留。

`restore_candidates_to_original()` 按 x/y 各自的实际缩放比例还原到原图。返回 `boxes [K,4]`、`scores [K]`、`labels [K]`，K 可以为 0 或大于 1。`save_prediction_image()` 只在绘图时取整，JSON 及评估保留浮点坐标。

批量预测入口先用 `collect_image_paths()` 收集显式指定的图片，或文件夹中的图片，再按 `--batch-size` 分组。`prepare_prediction_image()` 对每张图片完成相同的缩放、补边和尺寸记录，堆成图片批次后传给 `predict_batch()`。每张图独立使用自己的原图尺寸还原坐标，分别保存预览和 JSON；模型权重只加载一次。可选的 `--annotation-dir` 按图片名寻找 XML，仅用于叠加真实框。

## 如何判断检测效果

`evaluate_model()` 读取指定子集图片，复用 `predict_batch()`，再直接读取 XML 的原始坐标。每张图片、每个类别调用 `match_detections_for_class()`，按预测得分从高到低匹配。

匹配类别相同、IoU 最大且 IoU ≥ 0.5 的真实框。该框未被匹配时计为 TP，已匹配则计为重复 FP；IoU 不足或没有同类真实框也计为 FP。未匹配的真实目标数是 FN。

`summarize_class()` 合并所有图片中同一类别的预测，并再次按得分全局排序。不能先算每张图片 AP 再取平均。`average_precision()` 根据累计 TP/FP 得到 PR 曲线，取精确率包络并积分。四类 AP 的平均是 mAP50。

P/R 使用固定得分阈值，AP 使用低阈值候选的排序，两个指标回答的问题不同。验证指标用于训练和参数选择，测试集只用于最终报告。

## 模型权重与实验记录

`experiment.prepare_splits()` 固定数据成员并计算 XML 指纹；`load_checkpoint()` 检查权重文件中的类别映射，`load_model()` 加载网络供预测和评估，`save_json()` 保存结果。

训练入口每次创建新的模型和优化器，从第 1 轮开始训练。学习率使用带下限的指数衰减：`--lr` 是初始值，`--min-lr` 是下限，默认分别为 `1e-3` 和 `1e-6`；`--lr-decay` 是每轮乘数，默认 `0.95`。最低值仅作下限，训练结束时不要求达到它。

每轮参数更新完成后，为下一轮设置 `param_group["lr"] = max(min_lr, param_group["lr"] * lr_decay)`。第 e 轮使用的学习率为 `max(min_lr, initial_lr × lr_decay**(e-1))`；总训练轮数不参与该公式。触及下限后保持该值，不会回升。默认设置下，第 50 轮约为 `8.10e-5`，第 100 轮约为 `6.23e-6`；只训练一轮时使用初始值。

`history.json/csv` 中的 `lr` 记录该轮实际使用的学习率。`save_model()` 保存模型参数、类别定义、当前轮次、训练配置和数据划分，供预测和评估使用；训练配置记录 `scheduler="exponential"`、`lr`、`min_lr`、`lr_decay`。

训练入口每轮记录损失，在配置的轮次评估检测指标。`latest.pth` 保存最新一轮模型权重；`best.pth` 保存评估过的模型中 mAP50 最高者，同分时比较验证损失。`write_history()` 保存 JSON 和 CSV；`plot_history.py` 读取记录绘图。
