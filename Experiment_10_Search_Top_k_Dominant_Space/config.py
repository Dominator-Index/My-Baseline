import torch
import numpy as np
import os
# 设置 SwanLab API Key
os.environ["SWANLAB_API_KEY"] = "zrVzavwSxtY7Gs0GWo9xV"

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

num_layer = 4

# 超参数
training_config = {
    "np_seed": 12138,
    "torch_seed": 12138,
    "learning_rate": 1.00,
    "steps": 500,
    "record_steps": list(range(0, 501, 10)),
    "input_dim": 16,
    "hidden_dim": 32,
    "output_dim": 10,
    "variance": 0.01,
    "rank": 1,
    "top_k_pca_number": 100,
    "swanlab_project_name": "Baseline",
    "num_layer": num_layer,
    "eigenvalue_interval": 10,
    "method": "gap",
    "device": str(device),
    "swanlab_run_name": f"Experiment-10 for {num_layer}NN-rank{1}",
    "swanlab_api_key": "zrVzavwSxtY7Gs0GWo9xV"  # 添加到配置中
}