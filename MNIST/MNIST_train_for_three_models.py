# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import wandb
import plotting
from itertools import cycle
from Bulk_train_domaint_train import dominant_train, Bulk_train # 导入定义的函数

from MNIST_config import config, device
from MNIST_model import LinearNetwork, test_model
from MNIST_data_utils import load_mnist_data, _to_one_hot
from MNIST_hessian_utils import compute_hessian_eigen_pyhessian, compute_dominant_projection
from MNIST_check_dominant_space import Successive_Check_Dominant_Space, First_Last_Check_Dominant_Space

import sys
import os

# 获取 train.py 所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
nn_module_dir = os.path.join(parent_dir, '2 NN')
if nn_module_dir not in sys.path:
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
wandb.init(project=config["wandb_project_name"], name=config["wandb_run_name"])
wandb.config.update(config)

# --- 加载数据 ---
(train_loader, test_loader,
 X_train_full, Y_train_labels_full, Y_train_onehot_full,
 X_test_full, Y_test_labels_full, Y_test_onehot_full) = load_mnist_data(config, device)

# --- 定义损失函数 ---
loss_fn = nn.MSELoss()

# --- 初始化模型和优化器 ---
model_sgd = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], device).to(device)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=config["learning_rate"])

model_dominant = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], device).to(device)
optimizer_dominant = optim.SGD(model_dominant.parameters(), lr=config["learning_rate"])

model_bulk = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], device).to(device)
optimizer_bulk = optim.SGD(model_bulk.parameters(), lr=config["learning_rate"])

steps = config["steps"]
loss_history = {'SGD': {}, 'Dominant': {}, 'Bulk': {}}

# --- 训练控制标志和变量 ---
converged = False
convergence_step = -1
last_loss = float('inf')
initial_model_state = None

# --- 初始 SGD 训练直到收敛 (使用 model_sgd) ---
model_sgd.train()
data_iter = cycle(train_loader)

print("--- Initial SGD Training until Convergence ---")
for step in range(steps + 1):
    try:
        batch_X, batch_Y_labels = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch_X, batch_Y_labels = next(data_iter)

    batch_X, batch_Y_labels = batch_X.to(device), batch_Y_labels.to(device)
    batch_Y_onehot = _to_one_hot(batch_Y_labels, config["output_dim"], device)

    optimizer_sgd.zero_grad()
    y_predict_batch = model_sgd(batch_X)
    loss_batch = loss_fn(y_predict_batch, batch_Y_onehot)
    current_loss = loss_batch.item()
    loss_history['SGD'][step] = current_loss
    wandb.log({"loss_SGD": current_loss}, step=step)

    loss_batch.backward()
    optimizer_sgd.step()
    
    #if not converged and abs(last_loss - current_loss) <= 9e-7 and step > 0:
    if step > 15000 :
        converged = True
        convergence_step = step
        print(f"SGD converged at step {step}, loss: {current_loss:.6f}")
        initial_model_state = {name: param.clone().to(device) for name, param in model_sgd.named_parameters()}
        # 加载收敛时的模型状态到另外两个模型
        model_dominant.load_state_dict(initial_model_state)
        model_bulk.load_state_dict(initial_model_state)
        break # 停止初始 SGD 训练

    last_loss = current_loss

# --- 在收敛后分别训练三个模型 ---
if converged and convergence_step != -1:
    print("\n--- Continuing Training with Three Methods ---")

    data_iter_continued = cycle(train_loader)

    for step in range(convergence_step + 1, steps + 1):
        try:
            batch_X, batch_Y_labels = next(data_iter_continued)
        except StopIteration:
            data_iter_continued = iter(train_loader)
            batch_X, batch_Y_labels = next(data_iter_continued)

        batch_X, batch_Y_labels = batch_X.to(device), batch_Y_labels.to(device)
        batch_Y_onehot = _to_one_hot(batch_Y_labels, config["output_dim"], device)

        # --- Continued SGD ---
        optimizer_sgd.zero_grad()
        y_predict_sgd = model_sgd(batch_X)
        loss_sgd = loss_fn(y_predict_sgd, batch_Y_onehot)
        loss_history['SGD'][step] = loss_sgd.item()
        wandb.log({"loss_SGD": loss_sgd.item()}, step=step)
        loss_sgd.backward() 
        optimizer_sgd.step()
       
        # --- Dominant Train (使用 model_dominant 自身的梯度和 Hessian 特征向量) ---
        optimizer_dominant.zero_grad()
        y_predict_dominant = model_dominant(batch_X)
        loss_dominant = loss_fn(y_predict_dominant, batch_Y_onehot)
        loss_history['Dominant'][step] = loss_dominant.item()
        wandb.log({"loss_Dominant": loss_dominant.item()}, step=step)
        loss_dominant.backward(create_graph=True) # 计算 model_dominant 的梯度
        grads_full_dominant = torch.cat([param.grad.to(device).view(-1) for param in model_dominant.parameters() if param.grad is not None])

        eigenvalues_and_eigenvectors_dominant = compute_hessian_eigen_pyhessian(
            model_dominant, loss_fn, X_train_full, Y_train_onehot_full,
            top_k=config["top_k_pca_number"], device=device
        )
        top_eigenvectors_dominant = torch.from_numpy(eigenvalues_and_eigenvectors_dominant[1]).float().to(device)

        dominant_train(model_dominant, grads_full_dominant.to(device), top_eigenvectors_dominant, config["top_k_pca_number"], optimizer_dominant, loss_dominant, device)
        """
        # --- Bulk Train (使用 model_bulk 自身的梯度和 Hessian 特征向量) ---
        optimizer_bulk.zero_grad()
        y_predict_bulk = model_bulk(batch_X)
        loss_bulk = loss_fn(y_predict_bulk, batch_Y_onehot)
        loss_history['Bulk'][step] = loss_bulk.item()
        wandb.log({"loss_Bulk": loss_bulk.item()}, step=step)
        loss_bulk.backward(create_graph=True) # 计算 model_bulk 的梯度
        grads_full_bulk = torch.cat([param.grad.to(device).view(-1) for param in model_bulk.parameters() if param.grad is not None])

        eigenvalues_and_eigenvectors_bulk = compute_hessian_eigen_pyhessian(
            model_bulk, loss_fn, X_train_full, Y_train_onehot_full,
            top_k=config["top_k_pca_number"], device=device
        )
        top_eigenvectors_bulk = torch.from_numpy(eigenvalues_and_eigenvectors_bulk[1]).float().to(device)

        Bulk_train(model_bulk, grads_full_bulk.to(device), top_eigenvectors_bulk, config["top_k_pca_number"], optimizer_bulk, loss_bulk, device)
        """
# --- 训练结束后 ---
test_model(model_sgd, X_test_full, Y_test_labels_full, Y_test_onehot_full, device)
if model_dominant is not None:
    test_model(model_dominant, X_test_full, Y_test_labels_full, Y_test_onehot_full, device)
if model_bulk is not None:
    test_model(model_bulk, X_test_full, Y_test_labels_full, Y_test_onehot_full, device)

# --- 绘制 Loss 曲线 ---
plotting.plot_comparison_loss_with_phases(loss_history, convergence_step)

plotting.plot_loss_curve(loss_history['Dominant'], title="Dominant Training Loss Curve")
plotting.plot_loss_curve(loss_history['Bulk'], title="Bulk Training Loss Curve")
# --- 完成 wandb 运行 ---
wandb.finish()