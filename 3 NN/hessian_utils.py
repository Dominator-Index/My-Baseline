import torch
import numpy as np
from config_linear import device
from pyhessian import hessian
from pyhessian.hessian import hessian

# 计算 Hessian 矩阵的特征值和特征向量
def compute_hessian_eigen(loss, params, top_k = 832):  # previously top_k = 5
    params = list(params)
    
    # 获取所有参数的展平梯度
    grads_flat = []
    for p in params:
        g = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)[0]
        grads_flat.append(g.view(-1))  # 每个参数展平为向量
    grads_flat = torch.cat(grads_flat)  # 形状: (total_params,)
    total_params = grads_flat.size(0)   # 总参数量 p = 75
    
    # 计算Hessian矩阵
    hessian_rows = []
    for g in grads_flat:  # 遍历每个梯度元素
        # 计算二阶导数（Hessian行）
        hessian_row = torch.autograd.grad(
            outputs=g, 
            inputs=params, 
            retain_graph=True, 
            allow_unused=True
        )
        # 处理未使用的参数梯度（填充零）
        hessian_row_flat = []
        for h, p in zip(hessian_row, params):
            if h is None:
                h_flat = torch.zeros_like(p).view(-1)
            else:
                h_flat = h.view(-1)
            hessian_row_flat.append(h_flat)
        hessian_row_flat = torch.cat(hessian_row_flat)  # 形状: (total_params,)
        hessian_rows.append(hessian_row_flat)
    
    # 构建Hessian矩阵
    hessian_matrix = torch.stack(hessian_rows)  # 形状: (total_params, total_params)
    
    # 转换为NumPy并计算特征值/向量
    hessian_numpy = hessian_matrix.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(hessian_numpy)
    sorted_indices = np.argsort(-eigenvalues)  # 降序排列
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices][:, :top_k]

""""""
# This a function when running in remote, please check the input agruments before run.
# Function to compute the eigenvalues and eigenvectors of the Hessian matrix (using pyhessian)
def compute_hessian_eigen_pyhessian(model, criterion, data_loader, top_k=5, device=device):
    """
    Computes the top eigenvalues and eigenvectors of the Hessian matrix using the pyhessian library.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        data_loader: Data loader used for computing the Hessian.
        top_k: The number of top eigenvalues and eigenvectors to return.
        device: The computation device (CPU or CUDA).

    Returns:
        tuple: Contains two NumPy arrays:
            - eigenvalues: Top k eigenvalues in descending order (shape: (top_k,)).
            - eigenvectors: Corresponding eigenvectors (shape: (total_params, top_k)).
    """
    hessian_computer = hessian.Hessian(model=model, criterion=criterion, data_loader=data_loader, device=device)
    hessian_eigen = hessian_computer.eigenvalues(top_n=top_k)
    eigenvalues = np.array(hessian_eigen[0])
    eigenvectors = np.array(hessian_eigen[1])
    return eigenvalues, eigenvectors

def compute_layer_weight_eigenvalues(model, top_k=None):
    """
    计算模型每一层权重矩阵的特征值和特征向量
    
    Args:
        model: PyTorch模型
        top_k: 返回前k个特征值，如果为None则返回所有特征值
    
    Returns:
        dict: 包含每层特征值和特征向量的字典
    """
    layer_eigenvalues = {}
    layer_eigenvectors = {}
    
    for name, param in model.named_parameters():
        # 修改条件：匹配 W1, W2, W3 参数名，并且是2维矩阵
        if name in ['W1', 'W2', 'W3'] and param.dim() == 2:
            weight_matrix = param.detach().cpu().numpy()
            
            # 如果不是方阵，使用 W @ W.T 来计算特征值
            if weight_matrix.shape[0] != weight_matrix.shape[1]:
                # 对于非方阵，计算 W @ W.T 的特征值
                gram_matrix = np.dot(weight_matrix, weight_matrix.T)
                eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)
            else:
                # 对于方阵，直接计算特征值
                eigenvals, eigenvecs = np.linalg.eigh(weight_matrix)
            
            # 按降序排列
            sorted_indices = np.argsort(-eigenvals)
            eigenvals = eigenvals[sorted_indices]
            eigenvecs = eigenvecs[:, sorted_indices]
            
            # 如果指定了top_k，只返回前k个
            if top_k is not None:
                eigenvals = eigenvals[:top_k]
                eigenvecs = eigenvecs[:, :top_k]
            
            layer_eigenvalues[name] = eigenvals
            layer_eigenvectors[name] = eigenvecs
    
    return layer_eigenvalues, layer_eigenvectors

def compute_hessian_eigenvalues_pyhessian(model, criterion, data_loader, top_k=5, device='cuda'):
    """
    Compute only the top eigenvalues of the Hessian using pyhessian (without eigenvectors).

    Args:
        model: PyTorch model.
        criterion: Loss function.
        data_loader: Data loader used for computing the Hessian.
        top_k: Number of top eigenvalues to compute.
        device: Device to use.

    Returns:
        np.ndarray: Top-k eigenvalues.
    """
    hessian_computer = hessian(model=model, 
                                       criterion=criterion, 
                                       dataloader=data_loader, 
                                       )

    # 只返回特征值
    hessian_eigen= hessian_computer.eigenvalues(maxIter=1000, tol=1e-8, top_n=top_k, )
    eigenvalues = np.array(hessian_eigen[0])
    return eigenvalues

def compute_dominant_projection_matrix(top_eigenvectors, k):
    """
    根据特征向量构造投影矩阵 P_k = Σ u_i u_i^T
    Args:
        top_eigenvectors: 形状为 (p, k) 的矩阵，每列是特征向量
    Returns:
        P_k: 形状为 (p, p) 的投影矩阵
    """
    p = top_eigenvectors.shape[0]
    P_k = np.zeros((p, p))
    for i in range(k):
        u_i = top_eigenvectors[:, i].reshape(-1, 1)
        P_k += np.dot(u_i, u_i.T)  # 累加外积
    return torch.from_numpy(P_k).float().to(device)  # 转为Tensor并指定设备