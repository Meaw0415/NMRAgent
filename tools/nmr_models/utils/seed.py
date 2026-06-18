import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    固定所有随机种子以确保实验可复现。
    参数:
        seed (int): 随机种子值
        deterministic (bool): 是否启用确定性算法（会稍慢）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 控制 CuDNN 的随机性
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # 环境变量（有时 Dataloader / 多进程会用到）
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed fixed to {seed} | deterministic={deterministic}")
