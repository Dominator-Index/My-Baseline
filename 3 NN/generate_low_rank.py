import torch
from traditional_config import config, device
from config_linear import training_config
rank = training_config["rank"]

def generate_low_rank_matrix(input_dim, output_dim):
    """
    生成一个低秩矩阵，其前 rank × rank 部分为单位矩阵，其余为 0。

    Args:
        input_dim: 矩阵的输入维度（列数）
        output_dim: 矩阵的输出维度（行数）

    Returns:
        一个 output_dim × input_dim 的低秩PyTorch张量，秩为 rank
    """
    # 参数验证
    if input_dim < output_dim:
        raise ValueError(f"input_dim ({input_dim}) must be greater than or equal to output_dim ({output_dim})")
    
    # 检查rank是否合理
    max_possible_rank = min(input_dim, output_dim)
    if rank > max_possible_rank:
        print(f"⚠️  警告: rank ({rank}) 超过最大可能秩 ({max_possible_rank})，将使用 {max_possible_rank}")
        effective_rank = max_possible_rank
    else:
        effective_rank = rank
    
    # 生成低秩矩阵
    matrix = torch.zeros((output_dim, input_dim))
    identity_matrix = torch.eye(effective_rank)
    matrix[:effective_rank, :effective_rank] = identity_matrix
    
    # 验证生成的矩阵
    actual_rank = torch.linalg.matrix_rank(matrix).item()
    
    print(f"📊 生成低秩矩阵:")
    print(f"   形状: {matrix.shape}")
    print(f"   期望秩: {effective_rank}")
    print(f"   实际秩: {actual_rank}")
    print(f"   设备: {device}")
    
    return matrix.to(device)

def generate_low_rank_identity(input_dim, output_dim):
    """
    生成一个低秩恒等矩阵，前 output_dim × output_dim 部分为单位矩阵，其余为 0。
    （这个函数生成完整的恒等矩阵投影）
    """
    if input_dim < output_dim:
        raise ValueError(f"input_dim ({input_dim}) must be greater than or equal to output_dim ({output_dim})")
    
    matrix = torch.zeros((output_dim, input_dim))
    identity_matrix = torch.eye(output_dim)
    matrix[:output_dim, :output_dim] = identity_matrix
    
    actual_rank = torch.linalg.matrix_rank(matrix).item()
    
    print(f"📊 生成低秩恒等矩阵:")
    print(f"   形状: {matrix.shape}")
    print(f"   实际秩: {actual_rank}")
    
    return matrix.to(device)

def generative_dataset(input_dim, output_dim, use_custom_rank=False):
    """
    生成训练数据集
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        use_custom_rank: 是否使用自定义的rank（来自config）
    """
    x = torch.rand(input_dim, input_dim).to(device)
    
    if use_custom_rank:
        # 使用自定义rank的低秩矩阵
        projection_matrix = generate_low_rank_matrix(input_dim, output_dim)
        print(f"🎯 使用自定义rank={rank}的低秩矩阵作为目标")
    else:
        # 使用完整的低秩恒等矩阵
        projection_matrix = generate_low_rank_identity(input_dim, output_dim)
        print(f"🎯 使用完整的低秩恒等矩阵作为目标")
    
    y = torch.matmul(projection_matrix, x).to(device)
    
    print(f"📏 数据维度: x={x.shape}, y={y.shape}")
    
    return x, y

# 添加测试函数
def test_low_rank_matrices():
    """测试两种低秩矩阵生成函数"""
    print("\n🧪 测试低秩矩阵生成:")
    
    input_dim = config["input_dim"]   # 16
    output_dim = config["output_dim"] # 10
    
    print(f"\n1. 测试 generate_low_rank_matrix (rank={rank}):")
    matrix1 = generate_low_rank_matrix(input_dim, output_dim)
    
    print(f"\n2. 测试 generate_low_rank_identity:")
    matrix2 = generate_low_rank_identity(input_dim, output_dim)
    
    print(f"\n3. 矩阵对比:")
    print(f"   自定义rank矩阵的非零元素数: {torch.count_nonzero(matrix1)}")
    print(f"   恒等矩阵的非零元素数: {torch.count_nonzero(matrix2)}")
    
    return matrix1, matrix2

if __name__ == "__main__":
    test_low_rank_matrices()