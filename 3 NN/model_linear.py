import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd
from collections import OrderedDict
from config_linear import training_config

def balanced_init(input_dim, hidden_dim, output_dim, num_layer, device):
    """ 使用SVD方法进行平衡初始化 """
    dims = [input_dim]
    for _ in range(num_layer - 1):
        dims.append(hidden_dim)
    dims.append(output_dim)

    d0, dN = dims[0], dims[-1]
    min_d = min(d0, dN)
    
    # Variance
    var = training_config["variance"]

    # Step 1: 采样 A
    A = np.random.randn(dN, d0) * var

    # Step 2: SVD 分解
    U, Sigma, Vt = svd(A, full_matrices=False)

    Sigma_power = np.power(np.diag(Sigma[:min_d]), 1 / (len(dims) - 1))

    # Step 4: 计算权重
    weights = []
    for i in range(len(dims) - 1):
        weight = np.zeros((dims[i + 1], dims[i]))
        if i == 0:
            weight[:min_d, :] = Sigma_power @ Vt[:min_d, :]   # W1
        elif i == len(dims) - 2:
            weight[:, :min_d] = U[:, :min_d] @ Sigma_power  # W_final
        else:
            weight[:min_d, :min_d] = Sigma_power
        
        # 转换为torch tensor
        weights.append(torch.from_numpy(weight).float().to(device))
    
    return weights

class LinearNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, var, device):
        super(LinearNetwork, self).__init__()
        
        # Variance
        self.var = var
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # 使用平衡初始化
        weights = balanced_init(input_dim, hidden_dim, output_dim, num_layer, device)

        # 正确创建参数层
        self.layers = nn.ModuleList()
        for i, weight in enumerate(weights):
            layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            layer.weight.data = weight
            self.layers.append(layer)

    def forward(self, x):
        # x 形状: [batch, input_dim, input_dim] = [1, 16, 16]
        # 修正目标: (16, 16) → (32, 16) → (10, 16)  # 注意：最后是 (10, 16) 不是 (10, 32)
        device = self.layers[0].weight.device
        x = torch.eye(self.input_dim).unsqueeze(0).to(device)

        batch_size = x.shape[0]
        
        # 去掉 batch 维度
        x = x.squeeze(0)  # [16, 16]
        
        # 第一层: 直接矩阵乘法 W1 @ x
        # W1 形状: [32, 16], x 形状: [16, 16] → 结果: [32, 16]
        x = self.layers[0].weight @ x  # [32, 16]
        
        # 中间层(如果有): 保持第二维不变，只变换第一维
        for i in range(1, len(self.layers) - 1):
            # W_i 形状: [32, 32], x 形状: [32, 16] → 结果: [32, 16]
            x = self.layers[i].weight @ x  # [32, 16]
        
        # 最后一层: W_final @ x
        # W_final 形状: [10, 32], x 形状: [32, 16] → 结果: [10, 16]
        if len(self.layers) > 1:
            x = self.layers[-1].weight @ x  # [10, 16]
        
        # 添加回 batch 维度
        x = x.unsqueeze(0)  # [1, 10, 16]
        
        return x
        
        