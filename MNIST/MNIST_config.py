# config.py
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
np_seed = 0
torch_seed = 0

np.random.seed(np_seed)
torch.manual_seed(torch_seed)


# 超参数
config = {
    "np_seed": 0,
    "torch_seed": 0,
    "learning_rate": 0.01,
    "batch_size": 50, # 添加 Batch Size [Source: 1452 in 2405.16002v3.pdf]
    "steps": 20000, 
    #调整记录步数以适应更长的训练
    "record_steps": [100, 500, 1000, 3000, 5000, 10000, 15000, 20000],
    "input_dim": 784,
    "hidden_dim": 200,
    "output_dim": 10,
    "train_samples": 5000,
    "test_samples": 5000,
    "top_k_pca_number": 10, 
    "wandb_project_name": "MNIST",
    # Updated run name to reflect setup
    "wandb_run_name": "3 NN MLP+Tanh+SGD+MSE+MNIST-5k(tiny space set up)(Three models)"
}