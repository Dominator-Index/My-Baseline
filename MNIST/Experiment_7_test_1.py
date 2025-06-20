import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import wandb
import plotting 
import itertools
from itertools import cycle
import matplotlib.pyplot as plt

from MNIST_config import config, device
from MNIST_model import LinearNetwork, test_model
from MNIST_data_utils import load_mnist_data, _to_one_hot
from MNIST_hessian_utils import compute_hessian_eigen_pyhessian, compute_dominant_projection
from MNIST_check_dominant_space import Successive_Check_Dominant_Space, First_Last_Check_Dominant_Space
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from plotting import plot_cosine_similarity_all, plot_single_cosine_similarity, plot_single_projection_history, plot_projection_history_all

import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)
# 构建到 '2 NN' 文件夹的路径 (e.g., /path/to/parent/2 NN)
nn_module_dir = os.path.join(parent_dir, '2 NN')

if nn_module_dir not in sys.path: # 避免重复添加
    sys.path.append(nn_module_dir)
    
from PCA import Successive_Record_Steps_PCA, First_Last_Record_Steps_PCA
from COS_similarity import Successive_Record_Steps_COS_Similarity, First_Last_Record_Steps_COS_Similarity
from Compute_invariant_eigenvector import compute_invariant_matrix

# --- 设置随机种子 ---
np.random.seed(config["np_seed"])
torch.manual_seed(config["torch_seed"])
if device.type == 'cuda':
    torch.cuda.manual_seed_all(config["torch_seed"])

# --- 初始化 wandb ---
wandb.login(key="19f26ee33b3dd19e282387aa75e310e4b07df17a")   # Key API
wandb.init(project=config["wandb_project_name"], name=config["wandb_run_name"])
wandb.config.update(config)

# --- 初始化模型和优化器 ---
model = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], device).to(device)
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

top_k = 10
steps = config["steps"]
record_steps = config["record_steps"]
hessian_eigenvalues = {}
loss_full_history = {} # 用于记录 full loss
loss_batch_history = {} # 用于记录 batch loss
dominant_projection = {}
gradient_norms = {} # 用于记录 full batch gradient norm
update_matrix_norms = {} # 用于记录 full batch update norm
recorded_steps_top_eigenvectors = {}
recorded_steps_invariant_marix_w1 = {}
recorded_steps_invariant_marix_w2 = {}
train_accuracy_history = {} # 基于 full dataset

X_gradient_loss = {}

# Store batch and full gradients
batch_gradients = {}
full_gradients = {}

# Store batch and full gradients every step
batch_gradients_history = {}
full_gradients_history = {}

# Store cosine similarity between top eigenvector and gradient
batch_cos_similarity_history = {}
full_cos_similarity_history = {}
diff_cos_similarity_history = {}

# Store projection history
batch_projection_history = {}
full_projection_history = {}
diff_projection_history = {}

# Register hook for gradient computation
def register_hooks_gradients(model, grad_store, name_prefix=""):
    hooks = []
    for name, param in model.named_parameters():
        def save_grad_hook(name):   # 闭包传参
            def hook_fn(grad):
                grad_store[name] = grad.detach().clone()
            return hook_fn
        h = param.register_hook(save_grad_hook(name_prefix + name))
        hooks.append(h)
    return hooks

(train_loader, test_loader,
 X_train_full, Y_train_labels_full, Y_train_onehot_full,
 X_test_full, Y_test_labels_full, Y_test_onehot_full) = load_mnist_data(config, device)

def flatten_gradients(grad_dict):
    return torch.cat([g.view(-1) for g in grad_dict.values()])

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x  # already numpy


# --- 定义损失函数 ---
loss_fn = nn.MSELoss(reduction="mean")
data_iter = cycle(train_loader) # 使用 cycle 保证能迭代超过一个 epoch

# --- SGD 训练循环 ---
model.train()

print("Start training...")

for step in range(steps + 1):
    # Load batch data
    try:
        batch_X, batch_Y_labels = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch_X, batch_Y_labels = next(data_iter)
        
    batch_X, batch_Y_labels = batch_X.to(device), batch_Y_labels.to(device)
    batch_Y_onehot = _to_one_hot(batch_Y_labels, config["output_dim"], device)
    
    optimizer.zero_grad()
    
    if step % 50 != 0:
        # Forward and back propagation
        y_predict_batch = model(batch_X)
        loss_batch = loss_fn(y_predict_batch, batch_Y_onehot)
        loss_batch.backward()
    else: 
        # Register hook for batch_gradients computation
        batch_gradients.clear()
        batch_hooks = register_hooks_gradients(model, batch_gradients, name_prefix="batch_")

        # Forward and back propagation
        y_predict_batch = model(batch_X)
        loss_batch = loss_fn(y_predict_batch, batch_Y_onehot)
        loss_batch_history[step] = loss_batch.item()  # Batch Loss
        loss_batch.backward()
        
        # Flat batch gradients
        flat_batch_gradients = to_numpy(flatten_gradients(batch_gradients))
        
        # Remove hooks
        for h in batch_hooks:
            h.remove()
        
        # Compute full gradients
        model.zero_grad(set_to_none=True)
        
        full_gradients.clear()
        full_hooks = register_hooks_gradients(model, full_gradients, name_prefix="full_")
        
        y_predict_full = model(X_train_full)
        loss_full = loss_fn(y_predict_full, Y_train_onehot_full)
        loss_full_history[step] = loss_full.item()   # Full Loss
        loss_full.backward()
        
        for h in full_hooks:
            h.remove()
            
        # Flat full gradients
        flat_full_gradients = to_numpy(flatten_gradients(full_gradients))
        
        # Record gradients history batch & full
        batch_gradients_history[step] = flat_batch_gradients
        full_gradients_history[step] = flat_full_gradients
        
        # Compute Hessian eigenvalue and eigenvectors
        eigenvalues_and_eigenvectors = compute_hessian_eigen_pyhessian(
            model, loss_fn, X_train_full, Y_train_onehot_full,
            top_k= 2 * config["top_k_pca_number"], device=device
            )
        
        eigenvalues = eigenvalues_and_eigenvectors[0]
        top_eigenvectors = torch.from_numpy(eigenvalues_and_eigenvectors[1][:, :config["top_k_pca_number"]]).float().to(device) # 取前 top_k_pca_number 个特征向量
        
        # Compute cosine similarity for batch gradients
        first_eigenvector = top_eigenvectors[:, 0]  # Hessian's top eigenvector
        flat_batch_gradients = flat_batch_gradients.reshape(1, -1)  # Reshape to 2D
        first_eigenvector = first_eigenvector.reshape(1, -1)  # Reshape to 2D
        cosine_sim_batch = cosine_similarity(to_numpy(flat_batch_gradients), to_numpy(first_eigenvector))
        batch_cos_similarity_history[step] = cosine_sim_batch.item()
        print(f"Step {step}: Cosine similarity between batch gradient and Hessian's top eigenvector: {cosine_sim_batch.item()}")
        
        # Compute cosine similarity for full gradients
        flat_full_gradients = flat_full_gradients.reshape(1, -1)  # Reshape to 2D
        cosine_sim_full = cosine_similarity(to_numpy(flat_full_gradients), to_numpy(first_eigenvector))
        full_cos_similarity_history[step] = cosine_sim_full.item()
        print(f"Step {step}: Cosine similarity between full gradient and Hessian's top eigenvector: {cosine_sim_full.item()}")
        
        # Compute cosine similarity for gradient difference
        grad_diff = flat_full_gradients - flat_batch_gradients
        cosine_sim_diff = cosine_similarity(to_numpy(grad_diff), to_numpy(first_eigenvector))
        diff_cos_similarity_history[step] = cosine_sim_diff.item()
        print(f"Step {step}: Cosine similarity between gradient difference and Hessian's top eigenvector: {cosine_sim_diff.item()}")
        
        # Initialize projection vector
        batch_proj_vector = np.zeros_like(flat_batch_gradients)
        full_proj_vector = np.zeros_like(flat_full_gradients)
        diff_proj_vector = np.zeros_like(grad_diff)
        
        # Compute batch gradients projection
        for i in range(config["top_k_pca_number"]):
            batch_proj_vector += np.dot(to_numpy(top_eigenvectors[:, i]).T, to_numpy(flat_batch_gradients).flatten()) * to_numpy(top_eigenvectors[:, i])
        batch_projection = np.linalg.norm(batch_proj_vector) / np.linalg.norm(flat_batch_gradients)
        batch_projection_history[step] = batch_projection
        
        # Compute full gradients projection
        for i in range(config["top_k_pca_number"]):
            full_proj_vector += np.dot(to_numpy(top_eigenvectors[:, i]).T, to_numpy(flat_full_gradients).flatten()) * to_numpy(top_eigenvectors[:, i])
        full_projection = np.linalg.norm(full_proj_vector) / np.linalg.norm(flat_full_gradients)
        full_projection_history[step] = full_projection
        
        # Compute diff gradients projection
        for i in range(config["top_k_pca_number"]):
            diff_proj_vector += np.dot(to_numpy(top_eigenvectors[:, i]).T, to_numpy(grad_diff).flatten()) * to_numpy(top_eigenvectors[:, i])
        diff_projection = np.linalg.norm(diff_proj_vector) / np.linalg.norm(grad_diff)
        diff_projection_history[step] = diff_projection
        
        # Loss
        print(f"Step {step} finished: Loss: {loss_batch.item()}")
    
    optimizer.step()
    
print("Training Finished")

# Plot batch & full loss after training
print("Plot loss curve")
plotting.plot_loss_curve(loss_batch_history, title="Training batch loss curve") 
plotting.plot_loss_curve(loss_full_history, title="Training full loss curve")

# Plot cosine similarity after training
# 先逐条画图
print("Plotting cosine similarity...")
plot_single_cosine_similarity(batch_cos_similarity_history, interval=50, label="batch gradient", save_path="/home/ouyangzl/BaseLine/MNIST/images/MNIST_batch_cosine_similarity.png")
plot_single_cosine_similarity(full_cos_similarity_history, interval=50, label="full gradient", save_path="/home/ouyangzl/BaseLine/MNIST/images/MNIST_full_cosine_similarity.png")
plot_single_cosine_similarity(diff_cos_similarity_history, interval=50, label="gradient difference", save_path="/home/ouyangzl/BaseLine/MNIST/images/MNIST_diff_cosine_similarity.png")

# 然后统一画一张图
plot_cosine_similarity_all({
    "batch gradient": batch_cos_similarity_history,
    "full gradient": full_cos_similarity_history,
    "gradient difference": diff_cos_similarity_history,   
}, interval=50, save_path="/home/ouyangzl/BaseLine/MNIST/images/MNIST_all_cosine_similarity.png")

# Plot projection
# 先逐条画图
print("Plotting projection...")
plot_single_projection_history(batch_projection_history, interval=50, label="batch gradient", save_path="/home/ouyangzl/BaseLine/MNIST/images/MNIST_batch_projection.png")
plot_single_projection_history(full_projection_history, interval=50, label="full gradient", save_path="/home/ouyangzl/BaseLine/MNIST/images/MNIST_full_projection.png")
plot_single_projection_history(diff_projection_history, interval=50, label="gradient difference", save_path="/home/ouyangzl/BaseLine/MNIST/images/MNIST_diff_projection.png")

# 然后统一画一张图
plot_projection_history_all({
    "batch gradient": batch_projection_history,
    "full gradient": full_projection_history,
    "gradient difference": diff_projection_history,
    }, interval=50, save_path="/home/ouyangzl/BaseLine/MNIST/images/MNIST_all_projection.png")

print("Plotting Finished")

