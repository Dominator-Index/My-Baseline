import torch
from traditional_config import config, device

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
  return matrix.to(device)


def generative_dataset(input_dim, output_dim):
  
    x = torch.rand(input_dim, input_dim).to(device)
    projection_matrix = generate_low_rank_identity(input_dim, output_dim)
    y = torch.matmul(projection_matrix, x).to(device)

    return x, y