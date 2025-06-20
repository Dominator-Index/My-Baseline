import torch

def loss_fn(output, target):
    return 0.5 * torch.norm(output - target, p='fro')**2