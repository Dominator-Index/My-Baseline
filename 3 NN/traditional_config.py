import torch
import numpy as np
import wandb

# 定义设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 随机种子
np_seed = 12138
torch_seed = 12138

np.random.seed(np_seed)
torch.manual_seed(torch_seed)


# 超参数
config = {
    "np_seed": 12138,
    "torch_seed": 12138,
    "learning_rate": 1.0,
    "steps": 500,
    "record_steps": [5, 10, 15, 50, 100, 150, 200, 350, 500],
    "input_dim": 16,
    "hidden_dim": 32,
    "output_dim": 10,
    "variance": 0.01,   # 1856 是总参数数量
    "top_k_pca_number": 10,
    "rank": 1,
    "wandb_project_name": "Baseline",
    "wandb_run_name": "3NN+GD+without activate function+traditional loss" 
}