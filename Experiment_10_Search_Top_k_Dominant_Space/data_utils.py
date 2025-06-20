import torch
from config import config, device

def generate_low_rank_identity(input_dim,output_dim):
  """
  生成一个 PyTorch 张量，其前 m x m 部分为单位矩阵，其余为 0。

  Args:
    input_dim: 单位矩阵的维度，也决定了矩阵的“前 rank”。
    output_dim: 生成矩阵的完整输出维度。

  Returns:
    一个 PyTorch 张量。
  """
  if input_dim < output_dim:
    raise ValueError("input_dim must be greater than or equal to output_dim")
  matrix = torch.zeros((output_dim, input_dim))
  identity_matrix = torch.eye(output_dim)
  #Guassian = torch.randn(output_dim, output_dim)
  matrix[:output_dim, :output_dim] = identity_matrix
  #matrix[:output_dim, :output_dim] = Guassian

  """
  Normalization, considering W2W1 as W2W1I then I \in R^{input \times input}, therefore,
  when normalizating, we need to devide input_dim(or multiply 1/input_dim)
  """
  
  matrix = 1/input_dim * matrix
  return matrix.to(device)