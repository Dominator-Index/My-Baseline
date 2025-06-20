import numpy as np
import torch
import torch.nn as nn
from model_3nn import LinearNetwork as Model3NN
from model import LinearNetwork 

def verify_identity_equivalence():
    """验证当输入为identity时两个模型是否等价"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 使用设备: {device}")
    
    # 设置相同的随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建3 NN模型并移到设备
    print("🏗️ 创建3NN模型...")
    model_3nn = Model3NN(16, 32, 10, 0.01, device).to(device)
    
    # 重置随机种子（确保初始化相同）
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建Experiment_10模型 (num_layer=3) 并移到设备
    print("🏗️ 创建Experiment_10模型...")
    model_exp10 = LinearNetwork(16, 32, 10, 3, 0.01, device).to(device)
    
    # 检查模型是否在正确设备上
    print(f"🔍 Model3NN设备: {next(model_3nn.parameters()).device}")
    print(f"🔍 ModelExp10设备: {next(model_exp10.parameters()).device}")
    
    # 创建identity输入
    identity = torch.eye(16).unsqueeze(0).to(device)  # [1, 16, 16]
    print(f"🔍 Identity设备: {identity.device}")
    
    # 获取输出
    print("⚡ 计算3NN输出...")
    output_3nn = model_3nn(identity)      # [10, 16]
    print(f"3NN输出形状: {output_3nn.shape}, 设备: {output_3nn.device}")
    
    print("⚡ 计算Experiment_10输出...")
    output_exp10 = model_exp10(identity)  # [1, 10, 16]
    print(f"Exp10输出形状: {output_exp10.shape}, 设备: {output_exp10.device}")
    
    # 去掉batch维度进行比较
    output_exp10_squeezed = output_exp10.squeeze(0)  # [10, 16]
    
    # 确保都在同一设备上
    if output_3nn.device != output_exp10_squeezed.device:
        print(f"⚠️ 设备不匹配: {output_3nn.device} vs {output_exp10_squeezed.device}")
        output_3nn = output_3nn.to(device)
        output_exp10_squeezed = output_exp10_squeezed.to(device)
    
    # 计算差异
    diff = torch.abs(output_3nn - output_exp10_squeezed).max()
    print(f"🔍 输出差异: {diff.item()}")
    
    # 验证权重乘积
    print("🔍 验证权重乘积...")
    W_product_3nn = model_3nn.W3 @ model_3nn.W2 @ model_3nn.W1
    
    weights_exp10 = [layer.weight for layer in model_exp10.layers]
    W_product_exp10 = weights_exp10[2] @ weights_exp10[1] @ weights_exp10[0]
    
    # 确保权重乘积在同一设备上
    W_product_3nn = W_product_3nn.to(device)
    W_product_exp10 = W_product_exp10.to(device)
    
    weight_diff = torch.abs(W_product_3nn - W_product_exp10).max()
    print(f"🔍 权重乘积差异: {weight_diff.item()}")
    
    # 检查是否相等（允许数值误差）
    is_equal = diff < 1e-5 and weight_diff < 1e-5
    print(f"✅ 模型等价 (阈值1e-5): {is_equal}")
    
    # 显示详细信息
    print(f"\n📊 详细分析:")
    print(f"   输出差异: {diff.item():.2e}")
    print(f"   权重差异: {weight_diff.item():.2e}")
    print(f"   是否等价: {is_equal}")
    
    if not is_equal:
        print(f"\n🔍 进一步分析:")
        print(f"   3NN输出范围: [{output_3nn.min().item():.6f}, {output_3nn.max().item():.6f}]")
        print(f"   Exp10输出范围: [{output_exp10_squeezed.min().item():.6f}, {output_exp10_squeezed.max().item():.6f}]")
        
        # 检查是否只是数值精度问题
        is_close = torch.allclose(output_3nn, output_exp10_squeezed, rtol=1e-4, atol=1e-4)
        print(f"   数值近似相等 (rtol=1e-4): {is_close}")
    
    return is_equal

def debug_model_creation():
    """调试模型创建过程"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 设备: {device}")
    
    # 测试3NN模型创建
    print("\n🧪 测试3NN模型创建:")
    torch.manual_seed(42)
    np.random.seed(42)
    model_3nn = Model3NN(16, 32, 10, 0.01, device)
    print(f"   模型设备: {next(model_3nn.parameters()).device}")
    print(f"   W1形状: {model_3nn.W1.shape}")
    print(f"   W2形状: {model_3nn.W2.shape}")
    print(f"   W3形状: {model_3nn.W3.shape}")
    
    # 测试Experiment_10模型创建
    print("\n🧪 测试Experiment_10模型创建:")
    torch.manual_seed(42)
    np.random.seed(42)
    model_exp10 = LinearNetwork(16, 32, 10, 3, 0.01, device)
    print(f"   模型设备: {next(model_exp10.parameters()).device}")
    print(f"   层数: {len(model_exp10.layers)}")
    for i, layer in enumerate(model_exp10.layers):
        print(f"   Layer{i}形状: {layer.weight.shape}")

if __name__ == "__main__":
    print("🚀 开始模型验证")
    
    # 先调试模型创建
    debug_model_creation()
    
    print("\n" + "="*50)
    
    # 然后验证等价性
    try:
        verify_identity_equivalence()
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()