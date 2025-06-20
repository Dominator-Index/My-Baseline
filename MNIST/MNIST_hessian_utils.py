import torch
import numpy as np
from MNIST_config import device
from pyhessian import hessian


def compute_hessian_eigen_pyhessian(model, criterion, inputs, targets, top_k, device):
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
    hessian_computer = hessian(model, criterion, data=(inputs, targets), cuda=True)
    hessian_eigen = hessian_computer.eigenvalues(top_n=top_k)

    eigenvalues = np.array(hessian_eigen[0])

    eigenvectors_from_hessian = hessian_eigen[1]
    if isinstance(eigenvectors_from_hessian, list):
        eigenvectors_cpu = []
        for inner_list in eigenvectors_from_hessian:
            flattened_vector = torch.cat([torch.tensor(x).float().flatten().cpu() if not isinstance(x, torch.Tensor) else x.float().flatten().cpu() for x in inner_list])
            eigenvectors_cpu.append(flattened_vector.numpy())
        eigenvectors = np.array(eigenvectors_cpu)
    else:
        raise TypeError(f"Expected eigenvectors to be a list, but got {type(eigenvectors_from_hessian)}")

    return eigenvalues, eigenvectors.T

def compute_dominant_projection(top_eigenvectors, prev_top, k):
    """
    根据特征向量构造并应用投影操作: (\sum_{i=1}^{k} u_i u_i^T) @ prev_top (使用 PyTorch 张量)
    Args:
    top_eigenvectors: 形状为 (p, k) 的 PyTorch 张量，每列是特征向量
    k: 使用的顶部特征向量的数量
    prev_top: 形状为 (p,) 或 (p, m) 的 PyTorch 张量
    Returns:
    projection: 形状与 prev_top 相同的 PyTorch 张量，计算结果
    """
    p = top_eigenvectors.shape[0]
    projection = torch.zeros_like(prev_top, device=device)

    for i in range(k):
        u_i = top_eigenvectors[:, i].unsqueeze(1)
        projection += torch.matmul(u_i, torch.matmul(u_i.T, prev_top))

    return projection
