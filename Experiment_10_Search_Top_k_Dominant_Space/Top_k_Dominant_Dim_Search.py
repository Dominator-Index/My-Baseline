import torch
import numpy as np
import matplotlib.pyplot as plt
from hessian_utils import compute_hessian_eigenvalues_pyhessian, compute_hessian_eigen
import wandb

def find_dominant_subspace_dimension(eigenvalues, method='max_gap', min_k=1, max_k=None, gap_threshold=None, plot=False, title=""):
    """
    找到dominant子空间的维数k
    
    Args:
        eigenvalues: 降序排列的特征值数组
        method: 检测方法
            - 'max_gap': 寻找最大gap的位置
            - 'threshold_gap': 使用阈值检测gap
            - 'relative_gap': 使用相对gap检测
        min_k: 最小的k值
        max_k: 最大的k值，如果为None则使用特征值长度-1
        gap_threshold: gap阈值，仅在threshold_gap方法中使用
        plot: 是否绘制图像
        title: 图像标题
        
    Returns:
        k: dominant子空间的维数
        gaps: 所有的gap值
        gap_info: 详细的gap信息
    """
    eigenvalues = np.array(eigenvalues)
    n = len(eigenvalues)

    if max_k is None:
        max_k = n - 1
    
    max_k = min(max_k, n-1)  # 确保max_k不超过特征值长度-1

    # 计算相邻特征值的差
    gaps = np.diff(eigenvalues)  # gaps[i] = eigenvalues[i] - eigenvalues[i+1]
    gaps = -gaps   # 因为特征值是降序排列，所以取负数得到正的gap

    # 限制搜索范围
    valid_indices = np.arange(min_k - 1, min(max_k, len(gaps)))
    valid_gaps = gaps[valid_indices]

    if len(valid_gaps) == 0:
        return min_k, gaps, {"method": method, "k": min_k, "reason": "no valid range"}

    gap_info = {
        "method": method,
        "all_gaps": gaps.tolist(),
        "valid_indices": valid_indices.tolist(),
        "valid_gaps": valid_gaps.tolist(),
    }

    if method == 'max_gap':
        # 找到最大gap的位置
        max_gap_idx = np.argmax(valid_gaps)
        k = valid_indices[max_gap_idx] + 1  # +1因为gap[i]对应第i+1个特征值
        gap_info.update({
            "k": k,
            "max_gap": float(valid_gaps[max_gap_idx]),
            "max_gap_position": int(valid_indices[max_gap_idx])
        })
    
    elif method == 'threshold_gap':
        # 使用阈值检测gap
        if gap_threshold is None:
            gap_threshold = np.mean(valid_gaps) + 2 * np.std(valid_gaps)
        
        significant_gaps = valid_indices[valid_gaps > gap_threshold]
        if len(significant_gaps) == 0:
            k = significant_gaps[0] + 1
        else:
            k = min_k
        
        gap_info.update({
            "k": k,
            "gap_threshold": float(gap_threshold),
            "significant_gaps": significant_gaps.tolist()
        })
    
    elif method == "relative_gap":
        # 使用相对gap检测 (gap / eigenvalue)
        relative_gaps = valid_gaps / np.abs(eigenvalues[valid_indices])
        max_relative_gap_idx = np.argmax(relative_gaps)
        k = valid_indices[max_relative_gap_idx] + 1

        gap_info.update({
            "k": k,
            "relative_gaps": relative_gaps.tolist(),
            "max_relative_gap": float(relative_gaps[max_relative_gap_idx]),
            "max_relative_gap_position": int(valid_indices[max_relative_gap_idx])
        })
    
    else:
        raise ValueError(f"Unknown method: {method}")

    # 绘制图像
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 绘制特征值
        ax1.semilogy(range(1, len(eigenvalues) + 1), np.abs(eigenvalues), 'bo-', markersize=4)
        ax1.axvline(x=k, color='red', linestyle='--', alpha=0.7, label=f'k={k}')
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('Eigenvalue (log scale)')
        ax1.set_title(f'{title} - Eigenvalue Spectrum')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制gap
        gap_positions = np.arange(1, len(gaps) + 1)
        ax2.plot(gap_positions, gaps, 'go-', markersize=3, alpha=0.7, label='Gaps')
        ax2.axvline(x=k, color='red', linestyle='--', alpha=0.7, label=f'Selected k={k}')
        ax2.set_xlabel('Position (between λ_i and λ_(i+1))')
        ax2.set_ylabel('Gap = λ_i - λ_(i+1)')
        ax2.set_title(f'{title} - Eigenvalue Gaps')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 如果使用wandb，记录图像
        try:
            wandb.log({f"dominant_subspace_analysis_{title.replace(' ', '_')}": wandb.Image(fig)})
        except:
            pass
        
        plt.close(fig)
    
    return k, gaps, gap_info

def analyze_dominant_subspace_evolution(model, criterion, data_loader, selected_steps, 
                                      method='max_gap', device='cuda', plot_evolution=True):
    """
    分析训练过程中dominant子空间维数的演化
    
    Args:
        model: PyTorch模型
        criterion: 损失函数
        data_loader: 数据加载器
        selected_steps: 要分析的训练步数列表
        method: 检测方法
        device: 设备
        plot_evolution: 是否绘制演化图
        
    Returns:
        evolution_data: 包含每个步数的k值和相关信息的字典
    """
    evolution_data = {
        'steps': [],
        'k_values': [],
        'max_gaps': [],
        'eigenvalues_history': [],
        'gap_info_history': []
    }

    for step in selected_steps:
        print(f"Analyzing step {step}...")

        # 计算Hessian特征值
        eigenvalues = compute_hessian_eigenvalues_pyhessian(
            model=model,
            criterion=criterion, 
            data_loader=data_loader,
            top_k=min(100, sum(p.numel() for p in model.parameters())), # 计算足够多的特征值
            device=device
        )

        # 找到dominant子空间维数
        k, gaps, gap_info = find_dominant_subspace_dimension(
            eigenvalues, 
            method=method,
            plot=False,  # 单独分析时不绘图
            title=f"Step_{step}"
        )

        # 记录数据
        evolution_data['steps'].append(step)
        evolution_data['k_values'].append(k)
        evolution_data['max_gaps'].append(gap_info.get('max_gap', 0))
        evolution_data['eigenvalues_history'].append(eigenvalues.tolist())
        evolution_data['gap_info_history'].append(gap_info)

        print(f"Step {step}: k = {k}, max_gap = {gap_info.get('max_gap', 'N/A')}")

        # 绘制演化图
        if plot_evolution and len(evolution_data['steps']) > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制k值演化
        ax1.plot(evolution_data['steps'], evolution_data['k_values'], 'bo-', markersize=6)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Dominant Subspace Dimension (k)')
        ax1.set_title('Evolution of Dominant Subspace Dimension')
        ax1.grid(True, alpha=0.3)
        
        # 绘制最大gap演化
        ax2.plot(evolution_data['steps'], evolution_data['max_gaps'], 'ro-', markersize=6)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Maximum Gap')
        ax2.set_title('Evolution of Maximum Eigenvalue Gap')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 记录到wandb
        try:
            wandb.log({"dominant_subspace_evolution": wandb.Image(fig)})
        except:
            pass
        
        plt.close(fig)
    
    return evolution_data

def compare_methods(eigenvalues, title=""):
    """
    比较不同方法检测dominant子空间维数的结果
    
    Args:
        eigenvalues: 特征值数组
        title: 图像标题
        
    Returns:
        comparison_results: 不同方法的结果比较
    """
    methods = ['max_gap', 'threshold_gap', 'relative_gap']
    results = {}
    
    print(f"\n=== Comparing methods for {title} ===")
    
    for method in methods:
        k, gaps, gap_info = find_dominant_subspace_dimension(
            eigenvalues, 
            method=method,
            plot=False
        )
        results[method] = {'k': k, 'gap_info': gap_info}
        print(f"{method}: k = {k}")
    
    return results

# 示例使用函数
def example_usage():
    """
    示例：如何使用dominant子空间检测功能
    """
    # 生成示例特征值（实际使用时替换为真实的Hessian特征值）
    np.random.seed(42)
    
    # 创建一个有明显gap的特征值序列
    dominant_eigenvalues = np.linspace(10, 5, 5)  # 5个较大的特征值
    bulk_eigenvalues = np.linspace(1, 0.1, 20)   # 20个较小的特征值
    eigenvalues = np.concatenate([dominant_eigenvalues, bulk_eigenvalues])
    
    print("Example eigenvalues:", eigenvalues[:10])
    
    # 使用不同方法检测
    k_max_gap, gaps, gap_info = find_dominant_subspace_dimension(
        eigenvalues, 
        method='max_gap',
        plot=True,
        title="Example Analysis"
    )
    
    print(f"\nDetected dominant subspace dimension: k = {k_max_gap}")
    print(f"Gap information: {gap_info}")
    
    # 比较不同方法
    comparison = compare_methods(eigenvalues, "Example")
    
    return k_max_gap, gaps, gap_info, comparison

if __name__ == "__main__":
    # 运行示例
    example_usage()