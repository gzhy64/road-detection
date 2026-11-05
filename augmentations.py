"""对训练图片进行轻微光照和几何增强，并同步变换目标边框。"""

import torch
import torch.nn.functional as F


class TrainAugmentation:
    """在原图画布内增强样本，保留全部目标，随后仍按原流程处理为 416 输入。"""

    def __init__(self, probability=0.5, brightness=0.2, contrast=0.2,
                 scale_range=(0.9, 1.1), translate=0.05):
        if not 0 <= probability <= 1 or not 0 <= brightness <= 1 or not 0 <= contrast <= 1:
            raise ValueError("增强概率及亮度、对比度幅度须在 [0,1] 内。")
        if len(scale_range) != 2 or not 0 < scale_range[0] <= scale_range[1] < float("inf"):
            raise ValueError("缩放范围须为两个有限正数，且下界不大于上界。")
        if not 0 <= translate < 1:
            raise ValueError("平移比例须在 [0,1) 内。")
        self.probability = probability
        self.brightness = brightness
        self.contrast = contrast
        self.scale_range = scale_range
        self.translate = translate
        self.max_attempts = 10

    @staticmethod
    def _uniform(lower, upper):
        """使用 PyTorch 随机数，使主进程和 DataLoader 工作进程都沿用其种子。"""
        return lower + (upper - lower) * torch.rand(()).item()

    def _photometric(self, image):
        """调整亮度和围绕平均灰度的对比度，输出像素仍在 [0,1] 内。"""
        image = image * self._uniform(1 - self.brightness, 1 + self.brightness)
        weights = image.new_tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)
        mean_gray = (image * weights).sum(dim=0).mean()
        factor = self._uniform(1 - self.contrast, 1 + self.contrast)
        return ((image - mean_gray) * factor + mean_gray).clamp(0, 1)

    def _geometric(self, image, target):
        """围绕画布中心缩放并平移，仅接受全部标注框仍完整可见的变换。"""
        height, width = image.shape[-2:]
        boxes = target["boxes"]
        for _ in range(self.max_attempts):
            scale = self._uniform(*self.scale_range)
            offset_x = (1 - scale) * width / 2
            offset_y = (1 - scale) * height / 2
            new_boxes = boxes * scale
            new_boxes[:, 0::2] += offset_x
            new_boxes[:, 1::2] += offset_y

            min_dx, max_dx = -self.translate * width, self.translate * width
            min_dy, max_dy = -self.translate * height, self.translate * height
            if boxes.numel():
                # 限制可采样的平移区间，避免裁掉目标或丢弃小目标标注。
                min_dx = max(min_dx, -new_boxes[:, 0].min().item())
                max_dx = min(max_dx, width - new_boxes[:, 2].max().item())
                min_dy = max(min_dy, -new_boxes[:, 1].min().item())
                max_dy = min(max_dy, height - new_boxes[:, 3].max().item())
            if min_dx > max_dx or min_dy > max_dy:
                continue
            dx = self._uniform(min_dx, max_dx)
            dy = self._uniform(min_dy, max_dy)

            # grid_sample 使用从输出到输入的逆映射；与边框的正向变换保持一致。
            theta = image.new_tensor([
                [1 / scale, 0, -2 * dx / (scale * width)],
                [0, 1 / scale, -2 * dy / (scale * height)],
            ]).unsqueeze(0)
            batch = image.unsqueeze(0)
            grid = F.affine_grid(theta, batch.shape, align_corners=False)
            transformed = F.grid_sample(
                batch, grid, mode="bilinear", padding_mode="zeros", align_corners=False,
            ).squeeze(0)

            new_boxes[:, 0::2] = (new_boxes[:, 0::2] + dx).clamp(0, width)
            new_boxes[:, 1::2] = (new_boxes[:, 1::2] + dy).clamp(0, height)
            return transformed, dict(target, boxes=new_boxes)
        return image, target

    def __call__(self, image, target):
        """输入原图 float32 [3,H,W] 和原图坐标标注，返回同尺寸的增强样本。"""
        if torch.rand(()).item() < self.probability:
            image = self._photometric(image)
        if torch.rand(()).item() < self.probability:
            image, target = self._geometric(image, target)
        return image, target

    def to_config(self):
        """返回可写入实验配置及模型权重的实际增强参数。"""
        return {
            "enabled": True,
            "photometric_probability": self.probability,
            "geometric_probability": self.probability,
            "brightness_range": [1 - self.brightness, 1 + self.brightness],
            "contrast_range": [1 - self.contrast, 1 + self.contrast],
            "scale_range": list(self.scale_range),
            "translate_fraction": self.translate,
            "geometry_attempts": self.max_attempts,
            "keep_all_boxes": True,
            "fill_value": 0.0,
        }
