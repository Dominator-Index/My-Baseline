import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import os

class TrainingVisualizer:
    """训练过程可视化器，包含CSV保存功能"""
    
    def __init__(self, save_dir="./plots", max_eigenvalues=200, max_gaps=200):
        self.save_dir = save_dir
        self.max_eigenvalues = max_eigenvalues
        self.max_gaps = max_gaps
        
        # 存储数据
        self.steps = []
        self.eigenvalues_history = defaultdict(list)  # {step: [eigenvalues]}
        self.gaps_history = defaultdict(list)          # {step: [gaps]}
        self.loss_history = []
        self.top_k_history = []
        self.hessian_compute_times = []
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置matplotlib
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def update_data(self, step, eigenvalues, gaps, loss, top_k, hessian_time=None):
        """更新数据"""
        if step not in self.steps:
            self.steps.append(step)
        
        # 限制eigenvalues和gaps的数量
        eigenvalues = eigenvalues[:self.max_eigenvalues]
        gaps = gaps[:self.max_gaps]
        
        self.eigenvalues_history[step] = eigenvalues
        self.gaps_history[step] = gaps
        self.loss_history.append(loss)
        self.top_k_history.append(top_k)
        
        if hessian_time is not None:
            self.hessian_compute_times.append(hessian_time)
        else:
            self.hessian_compute_times.append(0.0)
    
    def save_data_to_csv(self, seed, num_layer):
        """保存所有数据到CSV文件"""
        if not self.steps:
            print("⚠️ 没有数据可保存")
            return
        
        print(f"💾 保存数据到CSV文件...")
        
        # 1. 保存基础统计数据
        self._save_basic_stats_csv(seed, num_layer)
        
        # 2. 保存特征值数据
        self._save_eigenvalues_csv(seed, num_layer)
        
        # 3. 保存间隙数据
        self._save_gaps_csv(seed, num_layer)
        
        # 4. 保存完整数据（宽格式）
        self._save_complete_data_csv(seed, num_layer)
        
        print(f"✅ 所有CSV文件已保存到: {self.save_dir}")
    
    def _save_basic_stats_csv(self, seed, num_layer):
        """保存基础统计数据"""
        data = {
            'step': sorted(self.steps),
            'loss': self.loss_history,
            'top_k_dominant_space': self.top_k_history,
            'hessian_compute_time': self.hessian_compute_times
        }
        
        # 添加统计指标
        max_eigenvalues = []
        min_eigenvalues = []
        condition_numbers = []
        eigenvalue_sums = []
        eigenvalue_means = []
        max_gaps = []
        gap_means = []
        
        for step in sorted(self.steps):
            eigenvals = self.eigenvalues_history[step]
            gaps = self.gaps_history[step]
            
            if eigenvals:
                max_eigenvalues.append(eigenvals[0])
                min_eigenvalues.append(eigenvals[-1])
                condition_numbers.append(eigenvals[0] / eigenvals[-1] if eigenvals[-1] != 0 else float('inf'))
                eigenvalue_sums.append(sum(eigenvals))
                eigenvalue_means.append(np.mean(eigenvals))
            else:
                max_eigenvalues.append(0)
                min_eigenvalues.append(0)
                condition_numbers.append(0)
                eigenvalue_sums.append(0)
                eigenvalue_means.append(0)
            
            if gaps:
                max_gaps.append(gaps[0])
                gap_means.append(np.mean(gaps[:200]))  # 前10个间隙的均值
            else:
                max_gaps.append(0)
                gap_means.append(0)
        
        data.update({
            'max_eigenvalue': max_eigenvalues,
            'min_eigenvalue': min_eigenvalues,
            'condition_number': condition_numbers,
            'eigenvalue_sum': eigenvalue_sums,
            'eigenvalue_mean': eigenvalue_means,
            'max_gap': max_gaps,
            'gap_mean': gap_means
        })
        
        df = pd.DataFrame(data)
        
        # 添加元数据
        df.attrs['seed'] = seed
        df.attrs['num_layer'] = num_layer
        
        csv_path = os.path.join(self.save_dir, f'basic_stats_seed{seed}_layer{num_layer}.csv')
        df.to_csv(csv_path, index=False)
        print(f"   📊 基础统计: {csv_path}")
    
    def _save_eigenvalues_csv(self, seed, num_layer):
        """保存特征值数据（长格式）"""
        eigenvalue_data = []
        
        for step in sorted(self.steps):
            eigenvals = self.eigenvalues_history[step]
            for i, eigenval in enumerate(eigenvals):
                eigenvalue_data.append({
                    'step': step,
                    'eigenvalue_index': i + 1,
                    'eigenvalue': eigenval,
                    'log_eigenvalue': np.log10(eigenval) if eigenval > 0 else -float('inf')
                })
        
        if eigenvalue_data:
            df = pd.DataFrame(eigenvalue_data)
            df.attrs['seed'] = seed
            df.attrs['num_layer'] = num_layer
            
            csv_path = os.path.join(self.save_dir, f'eigenvalues_seed{seed}_layer{num_layer}.csv')
            df.to_csv(csv_path, index=False)
            print(f"   🔢 特征值数据: {csv_path}")
        
        # 同时保存宽格式的特征值数据
        self._save_eigenvalues_wide_csv(seed, num_layer)
    
    def _save_eigenvalues_wide_csv(self, seed, num_layer):
        """保存特征值数据（宽格式，每列一个特征值）"""
        if not self.eigenvalues_history:
            return
        
        # 确定最大特征值数量
        max_eigenvals = max(len(eigenvals) for eigenvals in self.eigenvalues_history.values())
        
        data = {'step': sorted(self.steps)}
        
        # 为每个特征值创建一列
        for i in range(max_eigenvals):
            eigenval_series = []
            for step in sorted(self.steps):
                eigenvals = self.eigenvalues_history[step]
                if i < len(eigenvals):
                    eigenval_series.append(eigenvals[i])
                else:
                    eigenval_series.append(np.nan)
            data[f'eigenvalue_{i+1:03d}'] = eigenval_series
        
        df = pd.DataFrame(data)
        df.attrs['seed'] = seed
        df.attrs['num_layer'] = num_layer
        
        csv_path = os.path.join(self.save_dir, f'eigenvalues_wide_seed{seed}_layer{num_layer}.csv')
        df.to_csv(csv_path, index=False)
        print(f"   📈 特征值宽格式: {csv_path}")
    
    def _save_gaps_csv(self, seed, num_layer):
        """保存间隙数据"""
        gap_data = []
        
        for step in sorted(self.steps):
            gaps = self.gaps_history[step]
            for i, gap in enumerate(gaps):
                gap_data.append({
                    'step': step,
                    'gap_index': i + 1,
                    'gap': gap,
                    'log_gap': np.log10(gap) if gap > 0 else -float('inf')
                })
        
        if gap_data:
            df = pd.DataFrame(gap_data)
            df.attrs['seed'] = seed
            df.attrs['num_layer'] = num_layer
            
            csv_path = os.path.join(self.save_dir, f'gaps_seed{seed}_layer{num_layer}.csv')
            df.to_csv(csv_path, index=False)
            print(f"   📏 间隙数据: {csv_path}")
        
        # 同时保存宽格式的间隙数据
        self._save_gaps_wide_csv(seed, num_layer)
    
    def _save_gaps_wide_csv(self, seed, num_layer):
        """保存间隙数据（宽格式）"""
        if not self.gaps_history:
            return
        
        # 确定最大间隙数量
        max_gaps = max(len(gaps) for gaps in self.gaps_history.values() if gaps)
        
        if max_gaps == 0:
            return
        
        data = {'step': sorted(self.steps)}
        
        # 为每个间隙创建一列
        for i in range(max_gaps):
            gap_series = []
            for step in sorted(self.steps):
                gaps = self.gaps_history[step]
                if i < len(gaps):
                    gap_series.append(gaps[i])
                else:
                    gap_series.append(np.nan)
            data[f'gap_{i+1:03d}'] = gap_series
        
        df = pd.DataFrame(data)
        df.attrs['seed'] = seed
        df.attrs['num_layer'] = num_layer
        
        csv_path = os.path.join(self.save_dir, f'gaps_wide_seed{seed}_layer{num_layer}.csv')
        df.to_csv(csv_path, index=False)
        print(f"   📊 间隙宽格式: {csv_path}")
    
    def _save_complete_data_csv(self, seed, num_layer):
        """保存完整的实验元数据"""
        metadata = {
            'experiment_info': [
                f'seed_{seed}_layer_{num_layer}',
                f'total_steps_{len(self.steps)}',
                f'max_eigenvalues_{self.max_eigenvalues}',
                f'max_gaps_{self.max_gaps}',
                f'final_loss_{self.loss_history[-1] if self.loss_history else 0}',
                f'final_top_k_{self.top_k_history[-1] if self.top_k_history else 0}',
                f'total_hessian_time_{sum(self.hessian_compute_times):.2f}s'
            ]
        }
        
        df = pd.DataFrame(metadata)
        csv_path = os.path.join(self.save_dir, f'experiment_metadata_seed{seed}_layer{num_layer}.csv')
        df.to_csv(csv_path, index=False)
        print(f"   ℹ️  实验元数据: {csv_path}")
    
    def load_data_from_csv(self, seed, num_layer):
        """从CSV文件加载数据（用于后续分析）"""
        base_path = self.save_dir
        
        try:
            # 加载基础统计数据
            stats_path = os.path.join(base_path, f'basic_stats_seed{seed}_layer{num_layer}.csv')
            if os.path.exists(stats_path):
                stats_df = pd.read_csv(stats_path)
                self.steps = stats_df['step'].tolist()
                self.loss_history = stats_df['loss'].tolist()
                self.top_k_history = stats_df['top_k_dominant_space'].tolist()
                print(f"✅ 已加载基础统计数据: {stats_path}")
            
            # 加载特征值数据
            eigenvals_path = os.path.join(base_path, f'eigenvalues_wide_seed{seed}_layer{num_layer}.csv')
            if os.path.exists(eigenvals_path):
                eigenvals_df = pd.read_csv(eigenvals_path)
                for _, row in eigenvals_df.iterrows():
                    step = row['step']
                    eigenvals = [val for col, val in row.items() 
                               if col.startswith('eigenvalue_') and not pd.isna(val)]
                    self.eigenvalues_history[step] = eigenvals
                print(f"✅ 已加载特征值数据: {eigenvals_path}")
            
            # 加载间隙数据
            gaps_path = os.path.join(base_path, f'gaps_wide_seed{seed}_layer{num_layer}.csv')
            if os.path.exists(gaps_path):
                gaps_df = pd.read_csv(gaps_path)
                for _, row in gaps_df.iterrows():
                    step = row['step']
                    gaps = [val for col, val in row.items() 
                           if col.startswith('gap_') and not pd.isna(val)]
                    self.gaps_history[step] = gaps
                print(f"✅ 已加载间隙数据: {gaps_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False
    
    def plot_eigenvalues(self, seed, num_layer, save=True, show=False):
        """绘制特征值变化图"""
        if not self.steps:
            return
        
        plt.figure(figsize=(14, 8))
        
        # 准备数据
        steps_array = np.array(sorted(self.steps))
        
        # 获取最大特征值数量
        max_eigs = max(len(self.eigenvalues_history[step]) for step in steps_array)
        
        # 为每个特征值创建一条线
        colors = plt.cm.viridis(np.linspace(0, 1, min(max_eigs, self.max_eigenvalues)))
        
        for i in range(min(max_eigs, self.max_eigenvalues)):
            eigenval_series = []
            valid_steps = []
            
            for step in steps_array:
                eigenvals = self.eigenvalues_history[step]
                if i < len(eigenvals):
                    eigenval_series.append(eigenvals[i])
                    valid_steps.append(step)
            
            if eigenval_series:
                plt.plot(valid_steps, eigenval_series, 
                        color=colors[i], alpha=0.7, linewidth=1,
                        label=f'λ_{i+1}' if i < 200 else None)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Eigenvalue')
        plt.title(f'Eigenvalue Evolution - Seed {seed}, {num_layer} Layers')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 只显示前10个特征值的图例
        if max_eigs > 0:
            plt.legend(loc='upper right', ncol=2, fontsize=8)
        
        # 添加统计信息
        if self.eigenvalues_history:
            latest_step = max(self.steps)
            latest_eigenvals = self.eigenvalues_history[latest_step]
            if len(latest_eigenvals) > 0:
                plt.text(0.02, 0.98, 
                        f'Latest: λ_max={latest_eigenvals[0]:.2e}\n'
                        f'λ_min={latest_eigenvals[-1]:.2e}\n'
                        f'Condition={latest_eigenvals[0]/latest_eigenvals[-1]:.1e}',
                        transform=plt.gca().transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, f'eigenvalues_seed{seed}_layer{num_layer}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 特征值图保存至: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_gaps(self, seed, num_layer, save=True, show=False):
        """绘制间隙变化图"""
        if not self.steps:
            return
        
        plt.figure(figsize=(14, 8))
        
        # 准备数据
        steps_array = np.array(sorted(self.steps))
        
        # 获取最大间隙数量
        max_gaps = max(len(self.gaps_history[step]) for step in steps_array if self.gaps_history[step])
        
        if max_gaps == 0:
            print("⚠️ 没有间隙数据可绘制")
            plt.close()
            return
        
        # 为每个间隙创建一条线
        colors = plt.cm.plasma(np.linspace(0, 1, min(max_gaps, self.max_gaps)))
        
        for i in range(min(max_gaps, self.max_gaps)):
            gap_series = []
            valid_steps = []
            
            for step in steps_array:
                gaps = self.gaps_history[step]
                if i < len(gaps):
                    gap_series.append(gaps[i])
                    valid_steps.append(step)
            
            if gap_series:
                plt.plot(valid_steps, gap_series, 
                        color=colors[i], alpha=0.7, linewidth=1.5,
                        label=f'Gap_{i+1}' if i < 200 else None)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Gap Value')
        plt.title(f'Eigenvalue Gaps Evolution - Seed {seed}, {num_layer} Layers')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 只显示前10个间隙的图例
        if max_gaps > 0:
            plt.legend(loc='upper right', ncol=2, fontsize=8)
        
        # 添加统计信息
        if self.gaps_history:
            latest_step = max(self.steps)
            latest_gaps = self.gaps_history[latest_step]
            if len(latest_gaps) > 0:
                plt.text(0.02, 0.98, 
                        f'Latest: Gap_max={latest_gaps[0]:.2e}\n'
                        f'Gap_min={latest_gaps[-1]:.2e}\n'
                        f'Gap_ratio={latest_gaps[0]/latest_gaps[1]:.1f}' if len(latest_gaps) > 1 else '',
                        transform=plt.gca().transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, f'gaps_seed{seed}_layer{num_layer}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 间隙图保存至: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_summary(self, seed, num_layer, save=True, show=False):
        """绘制综合图表"""
        if not self.steps:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        steps_array = np.array(sorted(self.steps))
        
        # 1. 损失变化
        ax1.plot(steps_array, self.loss_history, 'b-', linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Top-k变化
        ax2.plot(steps_array, self.top_k_history, 'r-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Top-k Dominant Space')
        ax2.set_title('Dominant Space Size')
        ax2.grid(True, alpha=0.3)
        
        # 3. 最大特征值
        max_eigenvalues = [self.eigenvalues_history[step][0] if self.eigenvalues_history[step] 
                          else 0 for step in steps_array]
        ax3.plot(steps_array, max_eigenvalues, 'g-', linewidth=2)
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Max Eigenvalue')
        ax3.set_title('Maximum Eigenvalue')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. 最大间隙
        max_gaps = [self.gaps_history[step][0] if self.gaps_history[step] 
                   else 0 for step in steps_array]
        ax4.plot(steps_array, max_gaps, 'm-', linewidth=2)
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Max Gap')
        ax4.set_title('Maximum Gap')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Summary - Seed {seed}, {num_layer} Layers', fontsize=16)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, f'summary_seed{seed}_layer{num_layer}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 综合图保存至: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def clear_data(self):
        """清空数据，准备下一个实验"""
        self.steps.clear()
        self.eigenvalues_history.clear()
        self.gaps_history.clear()
        self.loss_history.clear()
        self.top_k_history.clear()