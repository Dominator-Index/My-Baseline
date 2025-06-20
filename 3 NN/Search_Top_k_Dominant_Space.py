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
import seaborn as sns
import sys
import os
import argparse
from hessian_utils import compute_hessian_eigenvalues_pyhessian, compute_hessian_eigen, compute_layer_weight_eigenvalues
from torch.utils.data import TensorDataset, DataLoader
from Top_k_Dominant_Dim_Search import find_dominant_subspace_dimension
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
    
wandb.login(key="zrVzavwSxtY7Gs0GWo9xV")
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

def plot_eigenvalue_evolution(eigenvalue_history, step_history, top_k, method, lr, var, seed):
    """绘制特征值演化图"""
    if len(step_history) == 0:
        print("⚠️  没有数据用于绘图")
        return
        
    plt.figure(figsize=(14, 10))
    
    # 设置颜色映射 - 使用更多颜色选项
    if top_k <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, top_k))
    elif top_k <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, top_k))
    else:
        # 对于更多特征值，使用渐变色
        colors = plt.cm.viridis(np.linspace(0, 1, top_k))
    
    # 绘制每个特征值的演化
    valid_lines = 0
    for i in range(min(len(eigenvalue_history), top_k)):
        key = f"top_{i+1}"
        if key in eigenvalue_history and len(eigenvalue_history[key]) > 0:
            # 确保数据长度一致
            data_len = len(eigenvalue_history[key])
            steps_for_this_data = step_history[:data_len]
            
            plt.plot(steps_for_this_data, 
                    eigenvalue_history[key], 
                    color=colors[i], 
                    linewidth=2 if i < 5 else 1.5,  # 前5个特征值用粗线
                    alpha=0.9 if i < 10 else 0.7,   # 前10个特征值更显眼
                    label=f'λ{i+1}',
                    marker='o' if len(steps_for_this_data) < 50 and i < 10 else None,
                    markersize=4 if i < 5 else 3)
            valid_lines += 1
    
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Eigenvalue', fontsize=14)
    plt.title(f'Evolution of Top {top_k} Hessian Eigenvalues\n{method} Optimizer, LR={lr}, Variance={var:.6f}, Seed={seed}', 
              fontsize=16)
    
    # 改进图例显示
    if valid_lines <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=1)
    else:
        # 对于太多特征值，只显示前20个的图例
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:20], labels[:20], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数尺度，更容易看到分叉
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图片
    plot_filename = f"eigenvalue_evolution_{method}_lr{lr}_var{var:.6f}_seed{seed}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 特征值演化图已保存: {plot_filename}")
    
    # 也记录到 wandb
    wandb.log({"Eigenvalue_Evolution_Plot": wandb.Image(plot_filename)})
    
    # 显示图片（如果在 Jupyter 中运行）
    plt.show()
    plt.close()

def plot_eigenvalue_divergence_analysis(eigenvalue_history, step_history, top_k, method, lr, var, seed):
    """分析特征值分叉点"""
    if len(step_history) < 2:
        print("⚠️  数据不足，无法进行分叉分析")
        return
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))
    
    # 设置颜色
    if top_k <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, top_k))
    elif top_k <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, top_k))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, top_k))
    
    # 上图：原始特征值演化 (显示前10个)
    display_count = min(10, top_k)
    for i in range(display_count):
        key = f"top_{i+1}"
        if key in eigenvalue_history and len(eigenvalue_history[key]) > 0:
            data_len = len(eigenvalue_history[key])
            steps_for_this_data = step_history[:data_len]
            ax1.plot(steps_for_this_data, 
                    eigenvalue_history[key], 
                    color=colors[i], 
                    linewidth=2, 
                    label=f'λ{i+1}')
    
    ax1.set_ylabel('Eigenvalue (log scale)')
    ax1.set_title(f'Top {display_count} Hessian Eigenvalue Evolution')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 中图：相邻特征值的比值（用于检测分叉）
    ratio_count = min(5, top_k-1)  # 显示前5个比值
    for i in range(ratio_count):
        key1 = f"top_{i+1}"
        key2 = f"top_{i+2}"
        if (key1 in eigenvalue_history and key2 in eigenvalue_history and 
            len(eigenvalue_history[key1]) > 0 and len(eigenvalue_history[key2]) > 0):
            
            # 计算比值
            min_len = min(len(eigenvalue_history[key1]), len(eigenvalue_history[key2]))
            ratios = [eigenvalue_history[key1][j] / max(eigenvalue_history[key2][j], 1e-10) 
                     for j in range(min_len)]
            
            ax2.plot(step_history[:min_len], ratios, 
                    color=colors[i], linewidth=2, 
                    label=f'λ{i+1}/λ{i+2}')
    
    ax2.set_ylabel('Eigenvalue Ratio')
    ax2.set_title('Eigenvalue Ratios (Divergence Detection)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 下图：特征值间隙分析
    gap_count = min(5, top_k-1)
    for i in range(gap_count):
        key1 = f"top_{i+1}"
        key2 = f"top_{i+2}"
        if (key1 in eigenvalue_history and key2 in eigenvalue_history and 
            len(eigenvalue_history[key1]) > 0 and len(eigenvalue_history[key2]) > 0):
            
            # 计算间隙
            min_len = min(len(eigenvalue_history[key1]), len(eigenvalue_history[key2]))
            gaps = [eigenvalue_history[key1][j] - eigenvalue_history[key2][j] 
                   for j in range(min_len)]
            
            ax3.plot(step_history[:min_len], gaps, 
                    color=colors[i], linewidth=2, 
                    label=f'λ{i+1} - λ{i+2}')
    
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Eigenvalue Gap')
    ax3.set_title('Eigenvalue Gaps (Absolute Differences)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存和记录
    plot_filename = f"eigenvalue_divergence_analysis_{method}_lr{lr}_var{var:.6f}_seed{seed}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    wandb.log({"Eigenvalue_Divergence_Analysis": wandb.Image(plot_filename)})
    
    print(f"📈 特征值分叉分析图已保存: {plot_filename}")
    plt.show()
    plt.close()

def plot_eigenvalue_spectrum_evolution(eigenvalue_history, step_history, top_k, method, lr, var, seed):
    """绘制特征值谱的演化热图"""
    if len(step_history) < 2:
        return
    
    # 准备数据矩阵
    data_matrix = []
    for step_idx, step in enumerate(step_history):
        eigenvals_at_step = []
        for i in range(top_k):
            key = f"top_{i+1}"
            if key in eigenvalue_history and step_idx < len(eigenvalue_history[key]):
                eigenvals_at_step.append(eigenvalue_history[key][step_idx])
            else:
                eigenvals_at_step.append(np.nan)
        data_matrix.append(eigenvals_at_step)
    
    data_matrix = np.array(data_matrix).T  # 转置，行为特征值，列为时间步
    
    # 创建热图
    plt.figure(figsize=(16, 8))
    
    # 使用对数变换
    log_data = np.log10(np.maximum(data_matrix, 1e-10))
    
    # 创建热图
    im = plt.imshow(log_data, cmap='viridis', aspect='auto', 
                   extent=[step_history[0], step_history[-1], top_k, 1])
    
    plt.colorbar(im, label='log₁₀(Eigenvalue)')
    plt.xlabel('Training Step')
    plt.ylabel('Eigenvalue Rank')
    plt.title(f'Hessian Eigenvalue Spectrum Evolution\n{method} Optimizer, LR={lr}, Variance={var:.6f}')
    
    # 设置y轴刻度
    plt.yticks(np.arange(1, min(top_k+1, 21), 2))
    
    plt.tight_layout()
    
    # 保存
    plot_filename = f"eigenvalue_spectrum_{method}_lr{lr}_var{var:.6f}_seed{seed}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    wandb.log({"Eigenvalue_Spectrum_Evolution": wandb.Image(plot_filename)})
    
    print(f"🌈 特征值谱演化图已保存: {plot_filename}")
    plt.show()
    plt.close()

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
    
    # 初始化数据收集变量
    eigenvalue_history = {f"top_{i+1}": [] for i in range(top_k)}
    step_history = []
    
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
                
                # 收集数据用于本地绘图
                step_history.append(step)
                for i, eigenval in enumerate(eigenvalues):
                    eigenvalue_history[f"top_{i+1}"].append(eigenval.item())
                
                # 使用分组记录特征值 - 这样会在同一张图中显示
                hessian_eigenvals = {}
                for i, eigenval in enumerate(eigenvalues):
                    # 使用 "/" 创建分组，这样会在同一张图中显示
                    hessian_eigenvals[f"Hessian_Eigenvalues/Top_{i+1}"] = eigenval.item()
                
                # 记录到 log_dict
                log_dict.update(hessian_eigenvals)
                
                # 也记录最大特征值（为了保持兼容性）
                log_dict[f"max_hessian_auto_{method}_{lr}_{var}_{seed}"] = eigenvalues[0].item()
                
            except Exception as e:
                tqdm.write(f"⚠️  Step {step}: 计算 Hessian 特征值失败: {e}")
            
            # 计算每一层权重矩阵的特征值 - 也使用分组
            try:
                layer_eigenvalues, layer_eigenvectors = compute_layer_weight_eigenvalues(model, top_k=1)
                
                # 使用分组记录层特征值
                layer_eigenvals = {}
                for layer_name, eigenvals in layer_eigenvalues.items():
                    max_eigenval = eigenvals[0]  # 最大特征值
                    layer_eigenvals[f"Layer_Eigenvalues/{layer_name}"] = max_eigenval
                
                # 记录到 log_dict
                log_dict.update(layer_eigenvals)
                    
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
    
    # 生成本地图表
    print(f"\n🎨 开始生成可视化图表...")
    if len(step_history) > 0:
        try:
            # 生成基本演化图
            plot_eigenvalue_evolution(eigenvalue_history, step_history, top_k, method, lr, var, seed)
            
            # 生成分叉分析图
            plot_eigenvalue_divergence_analysis(eigenvalue_history, step_history, top_k, method, lr, var, seed)
            
            # 生成谱演化热图
            plot_eigenvalue_spectrum_evolution(eigenvalue_history, step_history, top_k, method, lr, var, seed)
            
            print(f"✅ 所有可视化图表已生成完成!")
            
        except Exception as e:
            print(f"⚠️  生成图表时出错: {e}")
    else:
        print(f"⚠️  没有收集到特征值数据，无法生成图表")

# 完成训练
total_time = time.time() - total_start_time
print(f"\n🎉 所有训练完成! 总耗时: {format_time(total_time)}")
print(f"📝 结果已保存到 wandb: {wandb.run.url}")

# 完成 wandb 运行
wandb.finish()