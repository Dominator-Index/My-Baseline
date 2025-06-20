import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model_3nn import LinearNetwork as Model3NN
from model import LinearNetwork 
import matplotlib.pyplot as plt

def test_sgd_training_equivalence():
    """测试两个模型在SGD训练下是否保持等价"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 使用设备: {device}")
    
    # 训练参数
    learning_rate = 0.1
    num_steps = 100
    
    # 生成固定的训练数据（identity输入）
    torch.manual_seed(123)  # 固定数据生成种子
    identity_input = torch.eye(16).unsqueeze(0).to(device)  # [1, 16, 16]
    
    # 生成目标输出（低秩矩阵）
    rank = 3
    target_matrix = torch.zeros(10, 16).to(device)
    target_matrix[:rank, :rank] = torch.eye(rank).to(device)
    target_output = target_matrix.unsqueeze(0)  # [1, 10, 16]
    
    print(f"📊 训练数据:")
    print(f"   输入形状: {identity_input.shape}")
    print(f"   目标形状: {target_output.shape}")
    print(f"   目标rank: {torch.linalg.matrix_rank(target_output.squeeze(0)).item()}")
    
    # ========== 训练3NN模型 ==========
    print("\n🏗️ 创建并训练3NN模型...")
    torch.manual_seed(42)  # 固定模型初始化种子
    np.random.seed(42)
    
    model_3nn = Model3NN(16, 32, 10, 0.01, device).to(device)
    optimizer_3nn = optim.SGD(model_3nn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    losses_3nn = []
    outputs_3nn_history = []
    
    print("🚀 开始训练3NN模型...")
    for step in range(num_steps):
        optimizer_3nn.zero_grad()
        
        # 3NN模型的输出不需要输入
        output_3nn = model_3nn(identity_input)  # [10, 16]
        output_3nn = output_3nn.unsqueeze(0)    # [1, 10, 16] 匹配目标维度
        
        loss_3nn = criterion(output_3nn, target_output)
        loss_3nn.backward()
        optimizer_3nn.step()
        
        losses_3nn.append(loss_3nn.item())
        if step % 10 == 0:
            outputs_3nn_history.append(output_3nn.detach().clone())
            print(f"   Step {step:3d}: Loss = {loss_3nn.item():.6f}")
    
    final_output_3nn = output_3nn.detach()
    
    # ========== 训练Experiment_10模型 ==========
    print("\n🏗️ 创建并训练Experiment_10模型...")
    torch.manual_seed(42)  # 相同的模型初始化种子
    np.random.seed(42)
    
    model_exp10 = LinearNetwork(16, 32, 10, 3, 0.01, device).to(device)
    optimizer_exp10 = optim.SGD(model_exp10.parameters(), lr=learning_rate)
    
    losses_exp10 = []
    outputs_exp10_history = []
    
    print("🚀 开始训练Experiment_10模型...")
    for step in range(num_steps):
        optimizer_exp10.zero_grad()
        
        # Experiment_10模型需要输入
        output_exp10 = model_exp10(identity_input)  # [1, 10, 16]
        
        loss_exp10 = criterion(output_exp10, target_output)
        loss_exp10.backward()
        optimizer_exp10.step()
        
        losses_exp10.append(loss_exp10.item())
        if step % 10 == 0:
            outputs_exp10_history.append(output_exp10.detach().clone())
            print(f"   Step {step:3d}: Loss = {loss_exp10.item():.6f}")
    
    final_output_exp10 = output_exp10.detach()
    
    # ========== 比较结果 ==========
    print("\n📊 训练结果比较:")
    
    # 比较最终损失
    final_loss_diff = abs(losses_3nn[-1] - losses_exp10[-1])
    print(f"   最终损失差异: {final_loss_diff:.2e}")
    
    # 比较最终输出
    output_diff = torch.abs(final_output_3nn - final_output_exp10).max()
    print(f"   最终输出差异: {output_diff.item():.2e}")
    
    # 比较权重乘积
    W_product_3nn = model_3nn.W3 @ model_3nn.W2 @ model_3nn.W1
    weights_exp10 = [layer.weight for layer in model_exp10.layers]
    W_product_exp10 = weights_exp10[2] @ weights_exp10[1] @ weights_exp10[0]
    
    weight_diff = torch.abs(W_product_3nn - W_product_exp10).max()
    print(f"   权重乘积差异: {weight_diff.item():.2e}")
    
    # 判断是否等价
    is_equivalent = (final_loss_diff < 1e-6 and 
                    output_diff < 1e-6 and 
                    weight_diff < 1e-6)
    
    print(f"\n✅ 训练等价性: {is_equivalent}")
    
    if not is_equivalent:
        print(f"🔍 详细分析:")
        print(f"   损失等价: {final_loss_diff < 1e-6}")
        print(f"   输出等价: {output_diff < 1e-6}")
        print(f"   权重等价: {weight_diff < 1e-6}")
    
    # ========== 可视化 ==========
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(losses_3nn, 'b-', label='3NN Model', linewidth=2)
    plt.plot(losses_exp10, 'r--', label='Experiment_10 Model', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    # 损失差异
    plt.subplot(1, 3, 2)
    loss_differences = [abs(l1 - l2) for l1, l2 in zip(losses_3nn, losses_exp10)]
    plt.plot(loss_differences, 'g-', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('|Loss_3NN - Loss_Exp10|')
    plt.title('Loss Difference')
    plt.yscale('log')
    plt.grid(True)
    
    # 输出差异随时间变化
    plt.subplot(1, 3, 3)
    output_diffs = []
    for out_3nn, out_exp10 in zip(outputs_3nn_history, outputs_exp10_history):
        diff = torch.abs(out_3nn - out_exp10).max().item()
        output_diffs.append(diff)
    
    steps_recorded = list(range(0, num_steps, 10))
    plt.plot(steps_recorded, output_diffs, 'purple', marker='o', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Max Output Difference')
    plt.title('Output Difference Over Time')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 图表已保存为 'training_comparison.png'")
    
    return is_equivalent, {
        'losses_3nn': losses_3nn,
        'losses_exp10': losses_exp10,
        'final_output_diff': output_diff.item(),
        'final_weight_diff': weight_diff.item(),
        'final_loss_diff': final_loss_diff
    }

def test_different_inputs():
    """测试不同输入下的训练等价性"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🧪 测试不同输入类型:")
    
    # 测试不同类型的输入
    test_inputs = {
        'identity': torch.eye(16).unsqueeze(0),
        'random': torch.randn(1, 16, 16),
        'ones': torch.ones(1, 16, 16),
        'scaled_identity': 2.0 * torch.eye(16).unsqueeze(0)
    }
    
    for input_name, test_input in test_inputs.items():
        print(f"\n📋 测试输入类型: {input_name}")
        test_input = test_input.to(device)
        
        # 创建相同初始化的模型
        torch.manual_seed(42)
        np.random.seed(42)
        model_3nn = Model3NN(16, 32, 10, 0.01, device).to(device)
        
        torch.manual_seed(42)
        np.random.seed(42)
        model_exp10 = LinearNetwork(16, 32, 10, 3, 0.01, device).to(device)
        
        # 前向传播比较
        output_3nn = model_3nn(test_input)  # [10, 16]
        output_exp10 = model_exp10(test_input)  # [1, 10, 16]
        
        # 比较输出
        if input_name == 'identity':
            # identity情况下应该完全相等
            diff = torch.abs(output_3nn - output_exp10.squeeze(0)).max()
            expected_equal = True
        else:
            # 非identity情况下可能不同
            diff = torch.abs(output_3nn - output_exp10.squeeze(0)).max()
            expected_equal = False
        
        print(f"   输出差异: {diff.item():.2e}")
        print(f"   预期相等: {expected_equal}")
        print(f"   实际相等: {diff < 1e-6}")

if __name__ == "__main__":
    print("🚀 开始SGD训练等价性测试")
    
    # 主要训练测试
    is_equivalent, results = test_sgd_training_equivalence()
    
    print("\n" + "="*60)
    
    # 不同输入测试
    test_different_inputs()
    
    print("\n" + "="*60)
    print("🎯 测试总结:")
    print(f"   SGD训练等价性: {is_equivalent}")
    print(f"   最终输出差异: {results['final_output_diff']:.2e}")
    print(f"   最终权重差异: {results['final_weight_diff']:.2e}")
    print(f"   最终损失差异: {results['final_loss_diff']:.2e}")
    
    if is_equivalent:
        print("✅ 两个模型在SGD训练下完全等价！")
    else:
        print("⚠️ 两个模型在SGD训练下存在差异，需要进一步分析。")