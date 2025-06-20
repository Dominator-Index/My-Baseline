import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import wandb
from traditional_config import config, device

from model import LinearNetwork
from traditional_data_utils import generate_low_rank_identity, generative_dataset
from sam import SAM
import matplotlib.pyplot as plt  # Add matplotlib for plotting
import sys
import os
import argparse
from hessian_utils import compute_hessian_eigenvalues_pyhessian, compute_hessian_eigen, compute_layer_weight_eigenvalues
from torch.utils.data import TensorDataset, DataLoader

# Set arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with different optimizers")
    
    # Use standard optimizer (SGD, Adam, etc.)
    parser.add_argument('--use_optimizer', action='store_true', default=True, help='使用标准优化器')
    parser.add_argument('--use_sam', action='store_true',  help='使用SAM优化器')
    parser.add_argument('--use_adam', action='store_true', help='使用Adam优化器')
    parser.add_argument('--threshold', type=float, default=0.98, help='子空间相似度阈值')
    parser.add_argument('--rho', type=float, default=0.05, help='SAM的rho参数')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# 初始化模型和优化器
model = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], config["variance"], device).to(device)

# Print the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Learning Rate
lr = config["learning_rate"]
print(f"Learning rate is: {lr}")

# Variance 
var = model.var
print(f"Variance is: {var}")

# 获取 top_k 参数
top_k = config["top_k_pca_number"]
print(f"Computing top {top_k} Hessian eigenvalues")

# Choose optimizer based on arguments
if args.use_sam:
    # Use SAM optimizer
    base_optimizer = torch.optim.SGD # Defined for updating "sharpness-aware" 
    optimizer = SAM(model.parameters(), base_optimizer, lr=config["learning_rate"], momentum=0.9, rho=args.rho, adaptive=False)
    method = "SAM"
    print("Using SAM optimizer")
elif args.use_adam:
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    method = "Adam"
    print("Using Adam optimizer")
elif args.use_optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    method = "SGD"
    print("Using SGD optimizer")
else:
    # Default optimizer (you can choose a fallback)
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    
wandb.login(key="19f26ee33b3dd19e282387aa75e310e4b07df17a")
# 初始化 wandb
wandb.init(project=config["wandb_project_name"], name=config["wandb_run_name"])

# 使用 config 字典更新 wandb.config
wandb.config.update(config)

loss_function = nn.MSELoss(reduction='mean')

# 训练过程
steps = config["steps"]

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# seeds = [42, 100, 500, 2000, 12138]
seeds = [12138]  # For reproducibility, you can use a single seed or multiple seeds

for seed in seeds:
    # 设置随机种子
    set_seed(seed)
    
    # Generate fake data
    data, label = generative_dataset(config["input_dim"], config["output_dim"])
    # data, label 的原始形状就是 (input_dim, input_dim) 和 (output_dim, input_dim)
    # 只在最外层加一个 batch 维度
    data = data.unsqueeze(0).to(device)   # 变为 (1, input_dim, input_dim)
    label = label.unsqueeze(0).to(device) # 变为 (1, output_dim, input_dim)

    single_loader = DataLoader(
            TensorDataset(data, label),
            batch_size=1,
            shuffle=False
        )

    for step in range(steps + 1):
        output = model.forward(data)
        loss = loss_function(output, label)
        
        if args.use_sam:
            # First forward-backward pass
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second forward-backward pass
            output = model(data)
            loss = loss_function(output, label)
            loss.backward()

            optimizer.second_step(zero_grad=True)

        elif args.use_adam:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif args.use_optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
            with torch.no_grad():
                for param, grad in zip(model.parameters(), grads):
                    param.data.copy_(param - 0.01 * grad)  # 避免 in-place 修改
        
        print(f"Step {step}: Loss = {loss.item()}")
        
        # 准备记录字典，始终包含损失
        log_dict = {"Training Loss": loss.item()}
        
        # 每一步都计算 Hessian 的前 top_k 个特征值
        try:
            eigenvalues = compute_hessian_eigenvalues_pyhessian(
                model=model,
                criterion=loss_function,
                data_loader=single_loader,
                top_k=top_k,  # 使用配置文件中的 top_k_pca_number
                device=device
            )
            
            # 记录前 top_k 个特征值
            for i, eigenval in enumerate(eigenvalues):
                log_dict[f"hessian_eigenval_{i+1}_{method}_{lr}_{var}_{seed}"] = eigenval.item()
            
            # 也记录最大特征值（为了保持兼容性）
            log_dict[f"max_hessian_auto_{method}_{lr}_{var}_{seed}"] = eigenvalues[0].item()
            
        except Exception as e:
            print(f"Step {step}: 计算 Hessian 特征值失败: {e}")
            # 如果计算失败，至少记录损失
            pass
        
        # 计算每一层权重矩阵的特征值（保持原有功能）
        try:
            layer_eigenvalues, layer_eigenvectors = compute_layer_weight_eigenvalues(model, top_k=1)
            
            # 记录每一层的最大特征值
            for layer_name, eigenvals in layer_eigenvalues.items():
                max_eigenval = eigenvals[0]  # 最大特征值
                log_dict[f"max_eigenval_{layer_name}_{method}_{lr}_{var}_{seed}"] = max_eigenval
                
        except Exception as e:
            print(f"Step {step}: 计算层特征值失败: {e}")
        
        # 一次性记录所有指标
        wandb.log(log_dict, step=step)

# 完成 wandb 运行
wandb.finish()