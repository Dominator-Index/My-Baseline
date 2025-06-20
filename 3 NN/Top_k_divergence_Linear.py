import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import swanlab as wandb
import time
from datetime import datetime
from tqdm import tqdm
from config_linear import training_config as config
from config_linear import device

from model_linear import LinearNetwork
from generate_low_rank import generate_low_rank_matrix, generative_dataset
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
model = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], 3, config["variance"], device).to(device)

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
print(f"Computing top {top_k} Hessian eigenvalues (原始值，不归一化)")

# Get the rank from config
rank = config.get("rank", 5)  # 默认值为5
print(f"Rank is: {rank}")

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

# wandb.login(key="19f26ee33b3dd19e282387aa75e310e4b07df17a")
# 初始化 wandb
wandb.init(project=config["swanlab_project_name"], name=f"3NN+{method}+lr{lr}+var{var:.6f}_rank{rank}_top{top_k}_raw", api_key="zrVzavwSxtY7Gs0GWo9xV")

# 使用 config 字典更新 wandb.config
wandb.config.update(config)
wandb.config.update({
    "eigenvalue_interval": eigenvalue_interval, 
    "eigenvalue_type": "raw_unnormalized",
    "target_rank": rank,
    "matrix_type": "low_rank_identity"
})

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

def plot_training_loss(loss_history, step_history, method, lr, var, seed, rank):
    """绘制训练损失图"""
    if len(loss_history) == 0:
        print("⚠️  没有损失数据用于绘图")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 绘制损失曲线
    plt.plot(step_history, loss_history, 'b-', linewidth=2, label='Training Loss')
    
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'Training Loss Evolution\n{method} Optimizer, LR={lr}, Variance={var:.6f}, Rank={rank}, Seed={seed}', 
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
    plot_filename = os.path.join(IMAGE_SAVE_DIR, f"linear_training_loss_{method}_lr{lr}_var{var:.6f}_rank{rank}_seed{seed}_raw.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📉 训练损失图已保存: {plot_filename}")
    
    # 记录到 wandb
    wandb.log({"Training_Loss_Plot": wandb.Image(plot_filename)})
    
    # 如果在无GUI环境，注释掉plt.show()
    # plt.show()
    plt.close()

def plot_top_k_eigenvalues(eigenvalue_history, step_history, top_k, method, lr, var, seed, rank):
    """绘制前top k个原始特征值演化图（不归一化）"""
    if len(step_history) == 0:
        print("⚠️  没有特征值数据用于绘图")
        return
        
    plt.figure(figsize=(16, 12))  # 增大图片尺寸以容纳更多特征值
    
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
    
    # 计算特征值范围以便更好的可视化
    all_eigenvals = []
    for i in range(top_k):
        key = f"top_{i+1}"
        if key in eigenvalue_history and len(eigenvalue_history[key]) > 0:
            all_eigenvals.extend(eigenvalue_history[key])
    
    if all_eigenvals:
        max_eigenval = max(all_eigenvals)
        min_eigenval = min(all_eigenvals)
        eigenval_range = max_eigenval - min_eigenval
        print(f"📊 特征值范围: [{min_eigenval:.6f}, {max_eigenval:.6f}], 跨度: {eigenval_range:.6f}")
    
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
                'min': min(eigenvals),
                'range': max(eigenvals) - min(eigenvals)
            }
            
            # 绘制特征值，前几个用更显眼的样式
            plt.plot(steps_for_this_data, 
                    eigenvals, 
                    color=colors[i], 
                    linewidth=3 if i < 3 else (2.5 if i < 10 else 2),  # 前3个最粗，前10个较粗
                    alpha=0.9 if i < 10 else 0.7,   # 前10个特征值更显眼
                    label=f'λ{i+1} (原始)',
                    marker='o' if len(steps_for_this_data) < 30 and i < 5 else None,
                    markersize=6 if i < 3 else (5 if i < 10 else 4))
            valid_lines += 1
    
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Raw Hessian Eigenvalue (未归一化)', fontsize=14)
    plt.title(f'Evolution of Top {top_k} Raw Hessian Eigenvalues\n{method} Optimizer, LR={lr}, Variance={var:.6f}, Rank={rank}, Seed={seed}', 
          fontsize=16)
    
    # 改进图例显示
    if valid_lines <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        # 对于太多特征值，只显示前15个的图例
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:15], labels[:15], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.grid(True, alpha=0.3)
    
    # 对于原始特征值，先尝试线性尺度，如果跨度太大再用对数尺度
    if all_eigenvals and max(all_eigenvals) > 0:
        max_val = max(all_eigenvals)
        min_val = min([v for v in all_eigenvals if v > 0])  # 排除非正值
        if max_val / min_val > 1000:  # 如果跨度超过3个数量级，使用对数尺度
            plt.yscale('log')
            print("📊 使用对数尺度显示特征值（跨度较大）")
        else:
            print("📊 使用线性尺度显示特征值")
    
    # 添加详细统计信息
    if valid_lines > 0:
        # 将第221行改为：
        stats_text = f'Rank: {rank}\nEigenvalue Count: {valid_lines}\nComputations: {len(step_history)}'
        if valid_lines >= 1:
            first_ev = eigenvalue_stats.get('λ1', {})
            if first_ev:
                stats_text += f'\nλ1: {first_ev["initial"]:.6f} → {first_ev["final"]:.6f}'
                stats_text += f'\nλ1 Range: {first_ev["range"]:.6f}'
        
        if valid_lines >= 2:
            second_ev = eigenvalue_stats.get('λ2', {})
            if second_ev:
                stats_text += f'\nλ2: {second_ev["initial"]:.6f} → {second_ev["final"]:.6f}'
        
        plt.text(0.02, 0.02, stats_text, 
                transform=plt.gca().transAxes, fontsize=11, 
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    plot_filename = os.path.join(IMAGE_SAVE_DIR, f"linear_top{top_k}_eigenvalues_raw_{method}_lr{lr}_var{var:.6f}_rank{rank}_seed{seed}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Top-{top_k}原始特征值演化图已保存: {plot_filename}")
    
    # 记录到 wandb
    wandb.log({"Top_K_Raw_Eigenvalues_Plot": wandb.Image(plot_filename)})
    
    # 如果在无GUI环境，注释掉plt.show()
    # plt.show()
    plt.close()
    
    # 打印特征值统计信息
    print(f"\n📊 特征值统计信息:")
    for i, (name, stats) in enumerate(eigenvalue_stats.items()):
        if i < 5:  # 只打印前5个特征值的详细信息
            print(f"   {name}: 初始={stats['initial']:.6f}, 最终={stats['final']:.6f}, 变化={stats['range']:.6f}")

# 训练主循环
seeds = [12138]

# 记录总训练开始时间
total_start_time = time.time()
print(f"\n{'='*60}")
print(f"🚀 开始训练 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总步数: {steps+1}, Seeds: {seeds}")
print(f"特征值计算步骤: {len(selected_steps)} 个")
print(f"特征值类型: 原始值（不归一化）")
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
    
    # Generate fake data - 修复维度问题
    data, label = generative_dataset(config["input_dim"], config["output_dim"], use_custom_rank=True)
    
    # 在这里添加打印低秩矩阵的代码
    from generate_low_rank import generate_low_rank_matrix
    projection_matrix = generate_low_rank_matrix(config["input_dim"], config["output_dim"])
    print(f"\n📊 Low Rank Matrix (rank={config['rank']}):")
    print(projection_matrix)
    print(f"矩阵形状: {projection_matrix.shape}")
    print(f"实际秩: {torch.linalg.matrix_rank(projection_matrix)}")
    
    # 确保维度匹配 - 根据警告信息调整
    print(f"📏 原始数据维度: data={data.shape}, label={label.shape}")
    
    # 调整数据维度以匹配模型期望
    if data.dim() == 2:  # 如果数据是2维，添加batch维度
        data = data.unsqueeze(0)  # 添加batch维度
    if label.dim() == 2:  # 如果标签是2维，添加batch维度
        label = label.unsqueeze(0)  # 添加batch维度
    
    # 确保数据在正确的设备上
    data = data.to(device)
    label = label.to(device)
    
    print(f"📏 调整后数据维度: data={data.shape}, label={label.shape}")

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
        
        # 确保输出和标签维度匹配
        if output.shape != label.shape:
            print(f"⚠️  Step {step}: 维度不匹配 - output: {output.shape}, label: {label.shape}")
            # 尝试调整维度
            if output.dim() == 2 and label.dim() == 3:
                output = output.unsqueeze(0)
            elif output.dim() == 3 and label.dim() == 2:
                label = label.unsqueeze(0)
                
            # 如果调整后仍然不匹配，强制reshape
            if output.shape != label.shape:
                print(f"🔧 强制调整维度: {output.shape} -> {label.shape}")
                output = output.view(label.shape)
            
            # 最终验证
            if output.shape == label.shape:
                print(f"✅ 维度匹配成功: {output.shape}")
            else:
                print(f"❌ 维度匹配失败: output={output.shape}, label={label.shape}")
        
        loss = loss_function(output, label)
        # 只在特定步骤打印损失
        if step in [0, 1, 5, 10] or step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")
        
        # 收集所有步骤的损失
        step_history.append(step)
        loss_history.append(loss.item())
        
        # Backward pass and optimization
        if args.use_sam:
            loss.backward()
            optimizer.first_step(zero_grad=True)
            output = model(data)
            if output.shape != label.shape:
                if output.dim() == 2 and label.dim() == 3:
                    output = output.unsqueeze(0)
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
            # 手动梯度下降 - 避免内存泄漏警告
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data -= 0.01 * param.grad
        
        # 准备记录字典
        log_dict = {"Training Loss": loss.item()}
        
        # 特征值计算
        hessian_time = 0
        if step in selected_steps:
            hessian_start = time.time()
            
            # 记录特征值计算的步骤
            eigenvalue_step_history.append(step)
            
            # 计算 Hessian 的前 top_k 个特征值（原始值，不归一化）
            try:
                eigenvalues = compute_hessian_eigenvalues_pyhessian(
                    model=model,
                    criterion=loss_function,
                    data_loader=single_loader,
                    top_k=top_k,
                    device=device
                )
                
                # 确认这是原始特征值（pyhessian库默认返回原始值）
                print(f"📊 Step {step}: 前5个原始特征值 = {eigenvalues[:5]}")
                
                # 收集原始特征值用于本地绘图
                for i, eigenval in enumerate(eigenvalues):
                    # 直接使用原始特征值，不进行任何归一化
                    raw_eigenval = eigenval.item()  # 保持原始数值
                    eigenvalue_history[f"top_{i+1}"].append(raw_eigenval)
                
                # 使用分组记录原始特征值到 wandb
                hessian_eigenvals = {}
                for i, eigenval in enumerate(eigenvalues):
                    hessian_eigenvals[f"Raw_Hessian_Eigenvalues/Top_{i+1}"] = eigenval.item()
                
                log_dict.update(hessian_eigenvals)
                log_dict[f"max_raw_hessian_{method}_{lr}_{var}_{seed}"] = eigenvalues[0].item()
                
            except Exception as e:
                tqdm.write(f"⚠️  Step {step}: 计算 Hessian 特征值失败: {e}")
                import traceback
                tqdm.write(f"错误详情: {traceback.format_exc()}")
            
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
            plot_training_loss(loss_history, step_history, method, lr, var, seed, rank)
        
        # 2. Top-k原始特征值演化图
        if len(eigenvalue_step_history) > 0:
            plot_top_k_eigenvalues(eigenvalue_history, eigenvalue_step_history, top_k, method, lr, var, seed, rank)
        
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