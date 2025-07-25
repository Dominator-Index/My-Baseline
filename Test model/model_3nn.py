import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd
from traditional_config import config, device

def balanced_init(input_dim, hidden_dim, output_dim, device):
    """ 使用SVD方法进行平衡初始化 """
    dims = [input_dim, hidden_dim, hidden_dim, output_dim]
    d0, dN = dims[0], dims[-1]
    min_d = min(d0, dN)
    
    # Variance
    var = config["variance"]

    # Step 1: 采样 A
    A = np.random.randn(dN, d0) * var

    # Step 2: SVD 分解
    U, Sigma, Vt = svd(A, full_matrices=False)

    Sigma_power = np.power(np.diag(Sigma[:min_d]), 1 / (len(dims) - 1))

    # Step 4: 计算权重
    W1 = torch.zeros(hidden_dim, input_dim)
    W1[:min_d, :] = torch.from_numpy(Sigma_power @ Vt[:min_d, :]).float().to(device) 
    W2 = torch.zeros(hidden_dim, hidden_dim)
    W2[:min_d, :min_d] = torch.from_numpy(Sigma_power).float().to(device)# W2 ≃ Σ^(1/N)
    W3 = torch.zeros(output_dim, hidden_dim)
    W3[:, :min_d] = torch.from_numpy(U[:, :min_d] @ Sigma_power).float().to(device)
    
    return W1, W2, W3

class LinearNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, var, device):
        
        # Variance
        self.var = var
        
        super(LinearNetwork, self).__init__()

        # 使用平衡初始化
        W1_init, W2_init, W3_init = balanced_init(input_dim, hidden_dim, output_dim, device)

        self.W1 = nn.Parameter(W1_init)
        self.W2 = nn.Parameter(W2_init)
        self.W3 = nn.Parameter(W3_init)

    def forward(self, x = None):
        #return torch.matmul(self.W3, torch.matmul(self.W2, torch.matmul(self.W1, x))) (There is no input of x)
        return torch.matmul(self.W3, torch.matmul(self.W2, self.W1))
        