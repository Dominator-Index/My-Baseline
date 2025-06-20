# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import wandb
import plotting 
from itertools import cycle

from MNIST_config import config, device
from MNIST_model import LinearNetwork, test_model
from MNIST_data_utils import load_mnist_data, _to_one_hot
from MNIST_hessian_utils import compute_hessian_eigen_pyhessian, compute_dominant_projection
from MNIST_check_dominant_space import Successive_Check_Dominant_Space, First_Last_Check_Dominant_Space

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

steps = config["steps"]
record_steps = config["record_steps"]
hessian_eigenvalues = {}
loss_history = {} # 用于记录 full batch loss
dominant_projection = {}
gradient_norms = {} # 用于记录 full batch gradient norm
update_matrix_norms = {} # 用于记录 full batch update norm
recorded_steps_top_eigenvectors = {}
recorded_steps_invariant_marix_w1 = {}
recorded_steps_invariant_marix_w2 = {}
train_accuracy_history = {} # 基于 full dataset

X_gradient_loss = {}

(train_loader, test_loader,
 X_train_full, Y_train_labels_full, Y_train_onehot_full,
 X_test_full, Y_test_labels_full, Y_test_onehot_full) = load_mnist_data(config, device)

# --- 定义损失函数 ---
loss_fn = nn.MSELoss()

# --- SGD 训练循环 ---
model.train()
data_iter = cycle(train_loader) # 使用 cycle 保证能迭代超过一个 epoch

for step in range(steps + 1):
    try:
        batch_X, batch_Y_labels = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch_X, batch_Y_labels = next(data_iter)

    batch_X, batch_Y_labels = batch_X.to(device), batch_Y_labels.to(device)
    batch_Y_onehot = _to_one_hot(batch_Y_labels, config["output_dim"], device)

    optimizer.zero_grad()
    y_predict_batch = model(batch_X)
    loss_batch = loss_fn(y_predict_batch, batch_Y_onehot)

    loss_batch.backward()
    optimizer.step()


    # Obeserving more data points
    if step % 50 == 0:
        model.eval() # 切换到评估模式

        # 计算并记录 Full Batch Loss 和 Accuracy
        with torch.no_grad():
            y_predict_full = model(X_train_full)
            loss_full = loss_fn(y_predict_full, Y_train_onehot_full)
            current_full_loss = loss_full.item()
            loss_history[step] = current_full_loss 
            wandb.log({"loss": current_full_loss}, step=step) 

            _, predicted_indices_full = torch.max(y_predict_full, 1)
            accuracy_full = (predicted_indices_full == Y_train_labels_full).sum().item() / Y_train_labels_full.size(0)
            train_accuracy_history[step] = accuracy_full
            wandb.log({"train_accuracy": accuracy_full}, step=step) 

        # 计算 Full Batch Gradient 和相关指标
        for p in model.parameters(): p.requires_grad_(True)
        y_predict_full_for_grad = model(X_train_full)
        loss_full_for_grad = loss_fn(y_predict_full_for_grad, Y_train_onehot_full)
        grads_full = torch.autograd.grad(loss_full_for_grad, model.parameters(), create_graph=True, retain_graph=True)

        grad_flat_full = torch.cat([g.view(-1) for g in grads_full if g is not None])
        grad_norm_full = torch.norm(grad_flat_full).item()
        gradient_norms[step] = grad_norm_full 
        wandb.log({f"gradient_norm_step_{step}": grad_norm_full}, step=step)

        full_update_matrix = -config["learning_rate"] * grad_flat_full
        full_update_norm = torch.norm(full_update_matrix).item()
        update_matrix_norms[step] = full_update_norm 
        wandb.log({f"update_matrix_norm_step_{step}": full_update_norm}, step=step)

        # 计算 Hessian
        eigenvalues_and_eigenvectors = compute_hessian_eigen_pyhessian(
        model, loss_fn, X_train_full, Y_train_onehot_full,
        top_k= 2 * config["top_k_pca_number"], device=device
        )
        eigenvalues = eigenvalues_and_eigenvectors[0]
        top_eigenvectors = torch.from_numpy(eigenvalues_and_eigenvectors[1][:, :config["top_k_pca_number"]]).float().to(device) # 取前 top_k_pca_number 个特征向量

        hessian_eigenvalues[step] = eigenvalues
        recorded_steps_top_eigenvectors[step] = top_eigenvectors
        wandb.log({f"hessian_eigenvalues_step_{step}": wandb.Histogram(eigenvalues)}, step=step)

        # 计算投影
        projection = compute_dominant_projection(top_eigenvectors, grad_flat_full.to(device), config["top_k_pca_number"])
        dom_proj_norm = projection.norm().item()
        dominant_projection[step] = dom_proj_norm
        wandb.log({f"dominant_projection_norm_step_{step}": dom_proj_norm}, step=step)
        X_gradient_loss[step] = dom_proj_norm/grad_norm_full
        # 不变子空间分析
        with torch.no_grad():
            W1, W2 = None, None
            for name, param in model.named_parameters():
                if 'fc1.weight' in name: W1 = param.clone().detach().cpu()
                if 'fc3.weight' in name: W2 = param.clone().detach().cpu() # 根据你的函数调整

            if W1 is not None and W2 is not None:
                invariant_w1, invariant_w2 = compute_invariant_matrix(W1, W2)
                inv_w1_norm = np.linalg.norm(invariant_w1)
                inv_w2_norm = np.linalg.norm(invariant_w2)
                recorded_steps_invariant_marix_w1[step] = inv_w1_norm
                recorded_steps_invariant_marix_w2[step] = inv_w2_norm
                wandb.log({"invariant_w1_norm": inv_w1_norm}, step=step)
                wandb.log({"invariant_w2_norm": inv_w2_norm}, step=step)
            else:
                 recorded_steps_invariant_marix_w1[step] = np.nan
                 recorded_steps_invariant_marix_w2[step] = np.nan

        # 切回训练模式
        model.train()

# --- 训练结束后 ---
test_model(model, X_test_full, Y_test_labels_full, Y_test_onehot_full, device)

# --- 执行后续分析 (PCA, Cosine Similarity等) ---
successive_pca_spectrum = Successive_Record_Steps_PCA(recorded_steps_top_eigenvectors)
first_last_pca_spectrum = First_Last_Record_Steps_PCA(recorded_steps_top_eigenvectors)
successive_cos_similarity = Successive_Record_Steps_COS_Similarity(recorded_steps_top_eigenvectors)
first_last_cos_similarity = First_Last_Record_Steps_COS_Similarity(recorded_steps_top_eigenvectors)
successive_check_dominant_space = Successive_Check_Dominant_Space(recorded_steps_top_eigenvectors)
first_last_check_dominant_space = First_Last_Check_Dominant_Space(recorded_steps_top_eigenvectors)

plotting.plot_loss_curve(loss_history) 
plotting.plot_hessian_eigenvalues(hessian_eigenvalues)

"""
plotting.plot_cosine_similarity(successive_cos_similarity)
"""

plotting.plot_pca_spectrum(successive_pca_spectrum)

"""
plotting.plot_projection_norm(dominant_projection)
plotting.plot_gradient_norms(gradient_norms) 
plotting.plot_update_matrix_norms(update_matrix_norms)
plotting.plot_cosine_similarity_to_last(first_last_cos_similarity)
"""

plotting.plot_pca_top_k_eigenvectors(first_last_pca_spectrum)

"""
plotting.plot_successive_check(successive_check_dominant_space)
plotting.plot_first_last_check(first_last_check_dominant_space)
plotting.plot_invariant_matrix_norms(recorded_steps_invariant_marix_w1, title="W1")
plotting.plot_invariant_matrix_norms(recorded_steps_invariant_marix_w2, title="W2")
"""
plotting.plot_train_accuracy(train_accuracy_history)

plotting.plot_top_2k_eigenvalues(hessian_eigenvalues)
plotting.plot_X_loss(X_gradient_loss)
# --- 完成 wandb 运行 ---
wandb.finish()