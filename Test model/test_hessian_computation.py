import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

# 添加路径以导入hessian_utils
from hessian_utils import compute_hessian_eigen, compute_hessian_eigen_pyhessian

# 简单的测试网络
class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 8, bias=False)
        self.fc2 = nn.Linear(8, 2, bias=False)
        
        # 简单初始化
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def simple_hessian_test():
    """简单测试两种Hessian方法的差异"""
    print("🚀 开始简单Hessian对比测试")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型和数据
    model = SimpleNetwork().to(device)
    data = torch.randn(16, 4).to(device)
    targets = torch.randn(16, 2).to(device)
    criterion = nn.MSELoss()
    
    # 创建DataLoader (pyhessian需要)
    dataset = torch.utils.data.TensorDataset(data, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    
    print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    top_k = 5  # 只计算前5个特征值
    
    # 方法1: compute_hessian_eigen
    print("\n🔍 方法1: compute_hessian_eigen")
    try:
        start_time = time.time()
        
        # 计算loss并保持计算图
        output = model(data)
        loss = criterion(output, targets)
        
        eigenvals_1, eigenvecs_1 = compute_hessian_eigen(loss, model.parameters(), top_k=top_k)
        time_1 = time.time() - start_time
        
        print(f"   ✅ 成功! 用时: {time_1:.3f}s")
        print(f"   📈 前3个特征值: {eigenvals_1[:3]}")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        eigenvals_1, eigenvecs_1 = None, None
    
    # 方法2: compute_hessian_eigen_pyhessian  
    print("\n🔍 方法2: compute_hessian_eigen_pyhessian")
    try:
        start_time = time.time()
        
        eigenvals_2, eigenvecs_2 = compute_hessian_eigen_pyhessian(
            model, criterion, data_loader, top_k=top_k, device=device
        )
        time_2 = time.time() - start_time
        
        print(f"   ✅ 成功! 用时: {time_2:.3f}s")
        print(f"   📈 前3个特征值: {eigenvals_2[:3]}")
        
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        eigenvals_2, eigenvecs_2 = None, None
    
    # 比较结果
    if eigenvals_1 is not None and eigenvals_2 is not None:
        print(f"\n📊 结果比较:")
        print(f"   ⏱️ 速度: 方法1={time_1:.3f}s, 方法2={time_2:.3f}s")
        
        # 比较特征值
        diff = np.abs(eigenvals_1[:top_k] - eigenvals_2[:top_k])
        max_diff = np.max(diff)
        print(f"   📈 特征值最大差异: {max_diff:.2e}")
        
        print(f"   🔍 详细对比:")
        for i in range(min(top_k, len(eigenvals_1), len(eigenvals_2))):
            print(f"      λ{i+1}: {eigenvals_1[i]:.6f} vs {eigenvals_2[i]:.6f} (差异: {diff[i]:.2e})")
        
        # 简单可视化
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        indices = range(min(top_k, len(eigenvals_1), len(eigenvals_2)))
        plt.plot(indices, eigenvals_1[:len(indices)], 'bo-', label='Method 1', linewidth=2)
        plt.plot(indices, eigenvals_2[:len(indices)], 'ro--', label='Method 2', linewidth=2)
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalue Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(indices, diff[:len(indices)], 'go-', linewidth=2)
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('|Difference|')
        plt.title('Eigenvalue Differences')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('simple_hessian_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📈 对比图已保存")
        
    else:
        print(f"\n❌ 无法比较结果")

def training_step_comparison():
    """在训练过程中比较两种方法"""
    print(f"\n🏋️ 训练过程中的Hessian对比")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型和数据
    model = SimpleNetwork().to(device)
    data = torch.randn(16, 4).to(device)
    targets = torch.randn(16, 2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 创建DataLoader
    dataset = torch.utils.data.TensorDataset(data, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    
    top_k = 3
    
    print(f"📊 在前3个训练步骤中比较特征值:")
    
    for step in range(3):
        print(f"\n📍 训练步骤 {step + 1}:")
        
        # 前向传播
        output = model(data)
        loss = criterion(output, targets)
        
        print(f"   损失: {loss.item():.6f}")
        
        # 方法1
        try:
            eigenvals_1, _ = compute_hessian_eigen(loss, model.parameters(), top_k=top_k)
            print(f"   方法1特征值: {eigenvals_1[:top_k]}")
        except Exception as e:
            print(f"   方法1失败: {e}")
            eigenvals_1 = None
        
        # 方法2
        try:
            eigenvals_2, _ = compute_hessian_eigen_pyhessian(
                model, criterion, data_loader, top_k=top_k, device=device
            )
            print(f"   方法2特征值: {eigenvals_2[:top_k]}")
        except Exception as e:
            print(f"   方法2失败: {e}")
            eigenvals_2 = None
        
        # 比较
        if eigenvals_1 is not None and eigenvals_2 is not None:
            diff = np.abs(eigenvals_1[:top_k] - eigenvals_2[:top_k])
            print(f"   特征值差异: {diff}")
            print(f"   最大差异: {np.max(diff):.2e}")
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    # 运行简单测试
    simple_hessian_test()
    
    print("\n" + "="*50)
    
    # 运行训练步骤测试
    training_step_comparison()
    
    print(f"\n🎯 测试完成!")
    print(f"💡 如果PyHessian更慢，可能是因为:")
    print(f"   - 使用了更精确但慢的Lanczos算法")
    print(f"   - 包含更多数值稳定性检查")
    print(f"   - 避免计算完整Hessian矩阵，内存效率更高")