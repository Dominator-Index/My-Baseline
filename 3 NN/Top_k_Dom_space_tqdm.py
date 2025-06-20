import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import wandb
import time
from datetime import datetime
from tqdm import tqdm  # 进度条库
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
    
    # 新增间隔参数
    parser.add_argument('--eigenvalue_interval', type=int, default=10, 
                       help='计算特征值的间隔步数 (默认每10步计算一次)')
    
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

# 计算间隔设置
eigenvalue_interval = args.eigenvalue_interval
print(f"特征值计算间隔: 每 {eigenvalue_interval} 步计算一次")

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
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)  # 使用配置文件中的学习率
    method = "SGD"
    print("Using SGD optimizer")
else:
    # Default optimizer (you can choose a fallback)
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    method = "GD"
    
wandb.login(key="19f26ee33b3dd19e282387aa75e310e4b07df17a")
# 初始化 wandb
wandb.init(project=config["wandb_project_name"], name=f"3NN+GD+{lr}+{var}+{method}_top{top_k}",)

# 使用 config 字典更新 wandb.config
wandb.config.update(config)
wandb.config.update({"eigenvalue_interval": eigenvalue_interval})  # 记录间隔参数

loss_function = nn.MSELoss(reduction='mean')

# 训练过程
steps = config["steps"]

# 根据间隔动态生成计算特征值的步骤
important_early_steps = [1, 2, 3, 4, 5]  # 前几步很重要
interval_steps = list(range(0, steps + 1, eigenvalue_interval))  # 间隔步骤
final_steps = [steps] if steps not in interval_steps else []  # 确保包含最后一步

selected_steps = sorted(set(important_early_steps + interval_steps + final_steps))
print(f"将在以下步骤计算特征值: {selected_steps[:10]}{'...' if len(selected_steps) > 10 else ''}")
print(f"总共 {len(selected_steps)} 个计算点")

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"

# seeds = [42, 100, 500, 2000, 12138]
seeds = [12138]  # For reproducibility, you can use a single seed or multiple seeds

# 记录总训练开始时间
total_start_time = time.time()
print(f"\n{'='*60}")
print(f"🚀 开始训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总步数: {steps+1}, Seeds: {seeds}")
print(f"特征值计算步骤: {len(selected_steps)} 个")
print(f"{'='*60}\n")

for seed_idx, seed in enumerate(seeds):
    # 设置随机种子
    set_seed(seed)
    
    print(f"\n🌱 Seed {seed} ({seed_idx+1}/{len(seeds)})")
    
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

    # 创建进度条
    progress_bar = tqdm(range(steps + 1), 
                       desc=f"Seed {seed}", 
                       ncols=120,
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    # 时间统计变量
    step_times = []
    forward_times = []
    backward_times = []
    hessian_times = []
    logging_times = []

    for step in progress_bar:
        step_start_time = time.time()
        
        # Forward pass
        forward_start = time.time()
        output = model.forward(data)
        loss = loss_function(output, label)
        forward_time = time.time() - forward_start
        forward_times.append(forward_time)
        
        # Backward pass and optimization
        backward_start = time.time()
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
        backward_time = time.time() - backward_start
        backward_times.append(backward_time)
        
        # 准备记录字典，始终包含损失
        log_dict = {"Training Loss": loss.item()}
        
        # 特征值计算
        hessian_time = 0
        if step in selected_steps:
            hessian_start = time.time()
            
            # 计算 Hessian 的前 top_k 个特征值
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
                tqdm.write(f"⚠️  Step {step}: 计算 Hessian 特征值失败: {e}")
            
            # 计算每一层权重矩阵的特征值
            try:
                layer_eigenvalues, layer_eigenvectors = compute_layer_weight_eigenvalues(model, top_k=1)
                
                # 记录每一层的最大特征值
                for layer_name, eigenvals in layer_eigenvalues.items():
                    max_eigenval = eigenvals[0]  # 最大特征值
                    log_dict[f"max_eigenval_{layer_name}_{method}_{lr}_{var}_{seed}"] = max_eigenval
                    
            except Exception as e:
                tqdm.write(f"⚠️  Step {step}: 计算层特征值失败: {e}")
            
            hessian_time = time.time() - hessian_start
            hessian_times.append(hessian_time)
        
        # Logging
        logging_start = time.time()
        wandb.log(log_dict, step=step)
        logging_time = time.time() - logging_start
        logging_times.append(logging_time)
        
        # 计算单步总时间
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        
        # 更新进度条信息
        avg_step_time = np.mean(step_times[-10:])  # 最近10步的平均时间
        remaining_steps = steps - step
        eta = avg_step_time * remaining_steps
        
        # 显示信息
        postfix_dict = {
            'Loss': f'{loss.item():.4f}',
            'Time': f'{step_time:.2f}s'
        }
        
        if step in selected_steps:
            postfix_dict['EigenTime'] = f'{hessian_time:.1f}s'
        
        postfix_dict['ETA'] = format_time(eta)
        
        progress_bar.set_postfix(postfix_dict)
        
        # 每隔一定步数打印详细信息
        if step % 50 == 0 or step == steps or step in [1, 5, 10]:
            current_time = datetime.now().strftime('%H:%M:%S')
            eigenvalue_info = f" | EigenTime: {hessian_time:.2f}s" if step in selected_steps else ""
            tqdm.write(f"📊 Step {step:3d} | {current_time} | Loss: {loss.item():.6f} | Time: {step_time:.2f}s{eigenvalue_info}")
            
            if step > 0 and step % 50 == 0:
                total_elapsed = time.time() - total_start_time
                avg_hessian_time = np.mean(hessian_times) if hessian_times else 0
                tqdm.write(f"   📈 Avg Step: {np.mean(step_times):.2f}s | Avg Hessian: {avg_hessian_time:.2f}s | Total: {format_time(total_elapsed)}")

    # 训练完成后的统计信息
    progress_bar.close()
    
    print(f"\n✅ Seed {seed} 完成!")
    print(f"📊 时间统计:")
    print(f"   平均每步时间: {np.mean(step_times):.3f}s")
    print(f"   Forward平均: {np.mean(forward_times):.3f}s")
    print(f"   Backward平均: {np.mean(backward_times):.3f}s") 
    if hessian_times:
        print(f"   Hessian平均: {np.mean(hessian_times):.3f}s")
        print(f"   特征值计算次数: {len(hessian_times)}")
    print(f"   Logging平均: {np.mean(logging_times):.3f}s")
    print(f"   总训练时间: {format_time(sum(step_times))}")

# 完成训练
total_time = time.time() - total_start_time
print(f"\n🎉 所有训练完成! 总耗时: {format_time(total_time)}")
print(f"📝 结果已保存到 wandb: {wandb.run.url}")

# 完成 wandb 运行
wandb.finish()