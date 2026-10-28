"""
    检查 Python、PyTorch 和 CUDA 环境
"""

import sys

import torch


# 记录软件版本，方便后续排查环境问题和复现实验
print("Python 版本：", sys.version.split()[0])
print("PyTorch 版本：", torch.__version__)
print("PyTorch 构建使用的 CUDA 版本：", torch.version.cuda)

cuda_available = torch.cuda.is_available()
print("CUDA 是否可用：", cuda_available)

if cuda_available:
    properties = torch.cuda.get_device_properties(0)

    print("GPU 名称：", properties.name)
    print("GPU 总显存（GiB）：", round(
        properties.total_memory / (1024 ** 3), 2
    ))

    x = torch.ones((2, 2), dtype=torch.float32, device="cuda")
    # @ 表示矩阵乘法；实际计算用于检查 GPU 运算是否正常
    result = x @ x

    print("张量形状：", tuple(x.shape))
    print("张量数据类型：", x.dtype)
    print("张量设备：", x.device)

    # 将结果复制到 CPU，再转换为普通 Python 列表进行显示。
    print("GPU 计算结果：", result.cpu().tolist())
else:
    print("当前环境未检测到可用的 CUDA GPU。")