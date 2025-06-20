import torch
import numpy as np
import random
import os

def set_seeds(seed=42):
    """
    设置所有随机种子以确保实验可重现
    
    Args:
        seed (int): 随机种子值，默认为42
    """
    print(f"🌱 设置随机种子: {seed}")
    
    # Python 内置随机数
    random.seed(seed)
    
    # NumPy 随机数
    np.random.seed(seed)
    
    # PyTorch 随机数
    torch.manual_seed(seed)
    
    # CUDA 随机数（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        
        # 确保CUDA操作的确定性（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置环境变量（某些操作可能用到）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ 随机种子设置完成!")