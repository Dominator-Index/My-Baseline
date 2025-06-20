import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import wandb
import time
from datetime import datetime
from tqdm import tqdm
from traditional_config import config, device

from model import LinearNetwork
from traditional_data_utils import generate_low_rank_identity, generative_dataset
from sam import SAM
import matplotlib.pyplot as plt
import os
import argparse
from hessian_utils import compute_hessian_eigenvalues_pyhessian
from torch.utils.data import TensorDataset, DataLoader

# 创建图片保存文件夹
IMAGE_SAVE_DIR = "/home/ouyangzl/BaseLine/3 NN/images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
print(f"📁 图片将保存到: {IMAGE_SAVE_DIR}")

# Set arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with different optimizers")
    
    parser.add_argument('--use_optimizer', action='store_true', default=True, help='使用标准优化器')
    parser.add_argument('--use_sam', action='store_true', help='使用SAM优化器')
    parser.add_argument('--use_adam', action='store_true', help='使用Adam优化器')
    parser.add_argument('--rho', type=float, default=0.05, help='SAM的rho参数')
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
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=config["learning_rate"], momentum=0.9, rho=args.rho, adaptive=False)
    method = "SAM"
    print("Using SAM optimizer")
elif args.use_adam:
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    method = "Adam"
    print("Using Adam optimizer")
elif args.use_optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    method = "SGD"
    print("Using SGD optimizer")
else:
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    method = "GD"

wandb.login(key="19f26ee33b3dd19e282387aa75e310e4b07df17a")
# 初始化 wandb
wandb.init(project=config["wandb_project_name"], name=f"3NN+{method}+lr{lr}+var{var:.6f}_top{top_k}")

# 使用 config 字典更新 wandb.config
wandb.config.update(config)
wandb.config.update({"eigenvalue_interval": eigenvalue_interval})

loss_function = nn.MSELoss(reduction='mean')

# 训练过程
steps = config["steps"]

# 根据间隔动态生成计算特征值的步骤
important_early_steps = [1, 2, 3, 4, 5]
interval_steps = list(range(0, steps + 1, eigenvalue_interval))
final_steps = [steps] if steps not in interval_steps else []

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

def plot_training_loss(loss_history, step_history, method, lr, var, seed):
    """绘制训练损失图"""
    if len(loss_history) == 0:
        print("⚠️  没有损失数据用于绘图")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 绘制损失曲线
    plt.plot(step_history, loss_history, 'b-', linewidth=2, label='Training Loss')
    
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'Training Loss Evolution\n{method} Optimizer, LR={lr}, Variance={var:.6f}, Seed={seed}', 
              fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数尺度
    
    # 添加统计信息
    final_loss = loss_history[-1]
    min_loss = min(loss_history)
    initial_loss = loss_history[0]
    
    plt.text(0.02, 0.98, 
             f'Initial Loss: {initial_loss:.6f}\nFinal Loss: {final_loss:.6f}\nMin Loss: {min_loss:.6f}\nReduction: {(initial_loss-final_loss)/initial_loss*100:.2f}%', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    plot_filename = os.path.join(IMAGE_SAVE_DIR, f"training_loss_{method}_lr{lr}_var{var:.6f}_seed{seed}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📉 训练损失图已保存: {plot_filename}")
    
    # 记录到 wandb
    wandb.log({"Training_Loss_Plot": wandb.Image(plot_filename)})
    
    plt.show()
    plt.close()

def plot_top_k_eigenvalues(eigenvalue_history, step_history, top_k, method, lr, var, seed):
    """绘制前top k个特征值演化图"""
    if len(step_history) == 0:
        print("⚠️  没有特征值数据用于绘图")
        return
        
    plt.figure(figsize=(14, 10))
    
    # 设置颜色映射
    if top_k <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, top_k))
    elif top_k <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, top_k))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, top_k))
    
    # 绘制每个特征值的演化
    valid_lines = 0
    eigenvalue_stats = {}
    
    for i in range(top_k):
        key = f"top_{i+1}"
        if key in eigenvalue_history and len(eigenvalue_history[key]) > 0:
            # 确保数据长度一致
            data_len = len(eigenvalue_history[key])
            steps_for_this_data = step_history[:data_len]
            eigenvals = eigenvalue_history[key]
            
            # 统计信息
            eigenvalue_stats[f'λ{i+1}'] = {
                'initial': eigenvals[0],
                'final': eigenvals[-1],
                'max': max(eigenvals),
                'min': min(eigenvals)
            }
            
            plt.plot(steps_for_this_data, 
                    eigenvals, 
                    color=colors[i], 
                    linewidth=2.5 if i < 5 else 2,  # 前5个特征值用粗线
                    alpha=0.9 if i < 10 else 0.7,   # 前10个特征值更显眼
                    label=f'λ{i+1}',
                    marker='o' if len(steps_for_this_data) < 30 and i < 5 else None,
                    markersize=5 if i < 3 else 4)
            valid_lines += 1
    
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Eigenvalue', fontsize=14)
    plt.title(f'Evolution of Top {top_k} Hessian Eigenvalues\n{method} Optimizer, LR={lr}, Variance={var:.6f}, Seed={seed}', 
              fontsize=16)
    
    # 改进图例显示
    if valid_lines <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        # 对于太多特征值，只显示前15个的图例
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:15], labels[:15], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数尺度，更容易看到分叉
    
    # 添加统计信息
    if valid_lines > 0:
        stats_text = f'Steps: {len(step_history)}\nEigenvalues: {valid_lines}\nComputations: {len(step_history)}'
        if valid_lines >= 1:
            first_ev = eigenvalue_stats.get('λ1', {})
            if first_ev:
                stats_text += f'\nλ1: {first_ev["initial"]:.4f} → {first_ev["final"]:.4f}'
        
        plt.text(0.02, 0.02, stats_text, 
                transform=plt.gca().transAxes, fontsize=11, 
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    plot_filename = os.path.join(IMAGE_SAVE_DIR, f"top{top_k}_eigenvalues_{method}_lr{lr}_var{var:.6f}_seed{seed}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Top-{top_k}特征值演化图已保存: {plot_filename}")
    
    # 记录到 wandb
    wandb.log({"Top_K_Eigenvalues_Plot": wandb.Image(plot_filename)})
    
    plt.show()
    plt.close()

# 训练主循环
seeds = [12138]

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
    step_history = []  # 所有步骤（用于损失）
    eigenvalue_step_history = []  # 计算特征值的步骤
    loss_history = []  # 损失历史
    
    # Generate fake data
    data, label = generative_dataset(config["input_dim"], config["output_dim"])
    data = data.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)

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

    for step in progress_bar:
        step_start_time = time.time()
        
        # Forward pass
        output = model.forward(data)
        loss = loss_function(output, label)
        
        # 收集所有步骤的损失
        step_history.append(step)
        loss_history.append(loss.item())
        
        # Backward pass and optimization
        if args.use_sam:
            loss.backward()
            optimizer.first_step(zero_grad=True)
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
                    param.data.copy_(param - 0.01 * grad)
        
        # 准备记录字典
        log_dict = {"Training Loss": loss.item()}
        
        # 特征值计算
        hessian_time = 0
        if step in selected_steps:
            hessian_start = time.time()
            
            # 记录特征值计算的步骤
            eigenvalue_step_history.append(step)
            
            # 计算 Hessian 的前 top_k 个特征值
            try:
                eigenvalues = compute_hessian_eigenvalues_pyhessian(
                    model=model,
                    criterion=loss_function,
                    data_loader=single_loader,
                    top_k=top_k,
                    device=device
                )
                
                # 收集数据用于本地绘图
                for i, eigenval in enumerate(eigenvalues):
                    eigenvalue_history[f"top_{i+1}"].append(eigenval.item())
                
                # 使用分组记录特征值到 wandb
                hessian_eigenvals = {}
                for i, eigenval in enumerate(eigenvalues):
                    hessian_eigenvals[f"Hessian_Eigenvalues/Top_{i+1}"] = eigenval.item()
                
                log_dict.update(hessian_eigenvals)
                log_dict[f"max_hessian_auto_{method}_{lr}_{var}_{seed}"] = eigenvalues[0].item()
                
            except Exception as e:
                tqdm.write(f"⚠️  Step {step}: 计算 Hessian 特征值失败: {e}")
            
            hessian_time = time.time() - hessian_start
        
        # 记录到 wandb
        wandb.log(log_dict, step=step)
        
        # 计算单步总时间
        step_time = time.time() - step_start_time
        
        # 更新进度条信息
        postfix_dict = {
            'Loss': f'{loss.item():.4f}',
            'Time': f'{step_time:.2f}s'
        }
        
        if step in selected_steps:
            postfix_dict['EigenTime'] = f'{hessian_time:.1f}s'
        
        progress_bar.set_postfix(postfix_dict)
        
        # 定期打印信息
        if step % 50 == 0 or step == steps or step in [1, 5, 10]:
            current_time = datetime.now().strftime('%H:%M:%S')
            eigenvalue_info = f" | EigenTime: {hessian_time:.2f}s" if step in selected_steps else ""
            tqdm.write(f"📊 Step {step:3d} | {current_time} | Loss: {loss.item():.6f} | Time: {step_time:.2f}s{eigenvalue_info}")

    progress_bar.close()
    
    print(f"\n✅ Seed {seed} 完成!")
    
    # 生成图表
    print(f"\n🎨 开始生成可视化图表...")
    
    try:
        # 1. 训练损失图
        if len(loss_history) > 0:
            plot_training_loss(loss_history, step_history, method, lr, var, seed)
        
        # 2. Top-k特征值演化图
        if len(eigenvalue_step_history) > 0:
            plot_top_k_eigenvalues(eigenvalue_history, eigenvalue_step_history, top_k, method, lr, var, seed)
        
        print(f"✅ 所有可视化图表已生成完成!")
        print(f"📁 图片保存位置: {IMAGE_SAVE_DIR}")
        
    except Exception as e:
        print(f"⚠️  生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

# 完成训练
total_time = time.time() - total_start_time
print(f"\n🎉 所有训练完成! 总耗时: {format_time(total_time)}")
print(f"📝 结果已保存到 wandb: {wandb.run.url}")
print(f"📁 所有图片已保存到: {IMAGE_SAVE_DIR}")

# 完成 wandb 运行
wandb.finish()