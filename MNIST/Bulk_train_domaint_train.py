import torch
import numpy as np
import wandb
from MNIST_hessian_utils import compute_dominant_projection

def dominant_train(model, flattened_gradient, top_eigenvectors, k, optimizer, loss_batch, device):
    """
    使用梯度在主导特征向量子空间上的投影进行训练.
    """
    projection_flattened = compute_dominant_projection(top_eigenvectors, flattened_gradient, k)

    start_index = 0
    for name, param in model.named_parameters():
        num_elements = param.numel()
        end_index = start_index + num_elements
        param.grad = projection_flattened[start_index:end_index].reshape(param.shape).to(param.device)
        start_index = end_index

        """
        if param.grad is not None:
            print(f"Dominant Train - Parameter: {name}, Gradient (first 10 elements): {param.grad.flatten()[:10]}")
        else:
            print(f"Dominant Train - Parameter: {name}, Gradient: None")
        """

    optimizer.step()
    optimizer.zero_grad()

def Bulk_train(model, flattened_gradient, top_eigenvectors, k, optimizer, loss_batch, device):
    """
    使用梯度中与主导特征向量子空间正交的分量进行训练 .
    """
    projection_flattened = compute_dominant_projection(top_eigenvectors, flattened_gradient, k)
    orthogonal_component_flattened = flattened_gradient  - projection_flattened

    start_index = 0
    for name, param in model.named_parameters():
        num_elements = param.numel()
        end_index = start_index + num_elements
        param.grad = orthogonal_component_flattened[start_index:end_index].reshape(param.shape).to(param.device)
        start_index = end_index
    
    optimizer.step()
    optimizer.zero_grad()