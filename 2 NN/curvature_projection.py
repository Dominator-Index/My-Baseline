import torch
import numpy as np
from config import device
from pyhessian import hessian

def curvature_dominant_projection_matrix(eigenvalues, top_eigenvectors, top_k, device="cpu"):
    """
    构造矩阵 P_k = Σ λ_i u_i u_i^T
    Args:
        eigenvalues: shape (k,) 前 k 个特征值
        top_eigenvectors: shape (p, k) 特征向量矩阵，每列为一个特征向量
        top_k: 使用前 k 个特征对构建矩阵
    Returns:
        P_k: 形状为 (p, p) 的 torch.Tensor
    """
    assert eigenvalues.shape[0] >= top_k, "eigenvalues size mismatch"
    assert top_eigenvectors.shape[1] >= top_k, "top_eigenvectors size mismatch"
    
    p = top_eigenvectors.shape[0]
    P_k = np.zeros((p, p))
    for i in range(top_k):
        lambda_i = eigenvalues[i]
        u_i = top_eigenvectors[:, i].reshape(-1, 1)
        P_k += lambda_i * np.dot(u_i, u_i.T)
    return torch.from_numpy(P_k).float().to(device)


def curvature_projection_trajectory(recorded_steps_top_eigenvectors, recorded_steps_eigenvalues, top_k):
    """
    计算曲率投影轨迹
    Args:
        recorded_steps_top_eigenvectors: shape (T, p, k) 记录了每一步的前 k 个特征值和特征向量
        top_k: 投影矩阵 P_k 使用的特征数量
    Returns:
        curvature_projection_trajectory: shape (T, p, p) 曲率投影轨迹
    """
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
    trajectory = []
    num_steps = len(sorted_steps)
    
    for i in range(1, num_steps):
        prev_step = sorted_steps[i - 1]
        curr_step = sorted_steps[i]
        
        # 取前一步的最大特征向量
        v = recorded_steps_top_eigenvectors[prev_step][:, 0]  # shape: (p,)
        
        # 当前步的特征信息
        lambdas = recorded_steps_eigenvalues[curr_step]
        eigvecs = recorded_steps_top_eigenvectors[curr_step]  # shape: (p, k)
        
        # 构造 P_k
        P_k = curvature_dominant_projection_matrix(lambdas, eigvecs, top_k)
        P_k = P_k.cpu().numpy()  # convert to numpy if not already
        
        # 计算 P_k * v
        P_k_v = np.dot(P_k, v)
        
        # 计算 P_k * v 的 2 范数
        norm_P_k_v = np.linalg.norm(P_k_v)

        # 计算 v 的 2 范数
        norm_v = np.linalg.norm(v)
        
        curvature = norm_P_k_v / (norm_v + 1e-10)  # 防止除以0
        trajectory.append(curvature)
    
    return trajectory