import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import wandb
from traditional_config import config, device

from traditional_model import LinearNetwork
from traditional_data_utils import generate_low_rank_identity, generative_dataset
from Top_k_search import Top_Down
import json
import pickle

# To use the package in the previous folder
import sys
import os
import argparse

import plotting
from PCA import Successive_Record_Steps_PCA, First_Last_Record_Steps_PCA
from COS_similarity import Successive_Record_Steps_COS_Similarity, First_Last_Record_Steps_COS_Similarity
from check_dominant_space import Successive_Check_Dominant_Space, First_Last_Check_Dominant_Space
from Compute_invariant_eigenvector import compute_invariant_matrix

# Import SAM
# sys.path.append('/home/ouyangzl/sam')

from sam import SAM

# Get parent folder to import
current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

nn_dir = parent_dir

sys.path.append(nn_dir)

# Set arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model different optimizers")
    
    # Use standard optimizer (SGD, Adam, etc.)
    parser.add_argument('--use_optimizer', action='store_true', help='Flag to enable standard optimizer step')

    # Use SAM optimizer
    parser.add_argument('--use_sam', action='store_true', help='Flag to enable SAM optimizer')
    
    # Use Adam optimizer
    parser.add_argument('--use_adam', action='store_true', help='Flag to enable Adam optimizer')
    
    # Set threshold
    parser.add_argument('--threshold', type=float, default=0.98, help='Threshold for similarity between two subspaces')
    
    # Set Rho for SAM
    parser.add_argument('--rho', type=float, default=0.05, help='Rho for SAM')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Assign the value of threshold to tau
tau = args.threshold

# 初始化模型和优化器
model = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], device).to(device)

# Print the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Choose optimizer based on arguments
if args.use_sam:
    # Use SAM optimizer
    print("Using SAM optimizer")
    base_optimizer = torch.optim.SGD # Defined for updating "sharpness-aware" 
    optimizer = SAM(model.parameters(), base_optimizer, lr=config["learning_rate"], momentum=0.9, rho=args.rho, adaptive=False)
elif args.use_adam:
    print("Using Adam optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
elif args.use_optimizer:
    print("Using standard SGD optimizer")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
else:
    # Default optimizer (you can choose a fallback)
    print("No optimizer selected, using custom gradient updates")
    optimizer = optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    
from hessian_utils import compute_hessian_eigen, compute_dominant_projection_matrix, compute_hessian_eigen_pyhessian

# 设置随机种子
np.random.seed(config["np_seed"])
torch.manual_seed(config["torch_seed"])

wandb.login(key="19f26ee33b3dd19e282387aa75e310e4b07df17a")
# 初始化 wandb
wandb.init(project=config["wandb_project_name"], name=config["wandb_run_name"])

# 使用 config 字典更新 wandb.config
wandb.config.update(config)

loss_function = nn.MSELoss(reduction='mean')

# 训练过程
steps = config["steps"]
record_steps = config["record_steps"]
selected_steps = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
hessian_eigenvalues = {}
loss_history = []
dominant_projection = {}
gradient_norms = {}
update_matrix_norms = {}
record_top_eigenvectors = {} 
record_eigenvalues = {}
invariant_matrix_w1 = {}
invariant_matrix_w2 = {}
record_grads = {}
# top_k_trajectory = {}

def record_gradients(step, grads, record_dict):
    grads_cat = torch.cat([g.view(-1, 1) for g in grads], dim=0)
    grads_numpy = grads_cat.detach().cpu().numpy()
    record_dict[step] = grads_numpy
    
# Generate fake data
data, label = generative_dataset(config["input_dim"], config["output_dim"])

for step in range(steps + 1):
    output = model.forward(data)
    target = generate_low_rank_identity(config["input_dim"], config["output_dim"])
    loss = 1/2 * loss_function(output, label)
    loss_history.append(loss.item())
    
    if args.use_sam:
        # First forward-backward pass
        loss.backward()
        optimizer.first_step(zero_grad=False)

        # Second forward-backward pass
        output = model(data)
        loss = 1/2 * loss_function(output, label)
        loss.backward()

        grads = [p.grad for p in model.parameters()]
        record_gradients(step, grads, record_grads)

        optimizer.second_step(zero_grad=True)

    elif args.use_adam:
        optimizer.zero_grad()
        loss.backward()

        grads = [p.grad for p in model.parameters()]
        record_gradients(step, grads, record_grads)

        optimizer.step()

    elif args.use_optimizer:
        optimizer.zero_grad()
        loss.backward()

        grads = [p.grad for p in model.parameters()]
        record_gradients(step, grads, record_grads)

        optimizer.step()

    else:
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
        record_gradients(step, grads, record_grads)
        
        with torch.no_grad():
            for param, grad in zip(model.parameters(), grads):
                param.data.copy_(param - 0.01 * grad)  # 避免 in-place 修改
    
    """
    compute the number of grads and set different seed to generative seed
    
    num_grads = sum(1 for g in grads if g is not None)
    noise_seed_offset = step  
    """
    """
    if args.use_optimizer:
        loss.backward()
        optimizer.step()
    else:
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
        with torch.no_grad():   
            for param, grad in zip(model.parameters(), grads):
                
                
                current_rng_state = torch.get_rng_state()
                current_np_state = np.random.get_state() 

                torch.manual_seed(config["torch_seed"] + noise_seed_offset)
                np.random.seed(config["np_seed"] + noise_seed_offset) 

                noise = torch.randn_like(grad)/num_grads
                grad = grad + noise
                

                param.data.copy_(param - 0.01 * grad)  # 避免 in-place 修改
    
                
                torch.set_rng_state(current_rng_state)
                np.random.set_state(current_np_state)
    """            
    if step in selected_steps:
        wandb.log({"loss": loss.item()}, step=step)

    # if step in record_steps:
    W1 = None
    W2 = None
    for name, param in model.named_parameters():
        if name == 'W1':
            W1 = param.data.clone().detach().cpu()  
        elif name == 'W2':
            W2 = param.data.clone().detach().cpu()
    
    """
    noisy_analysis_grads = []
    for grad in grads:
            current_rng_state = torch.get_rng_state()
            current_np_state = np.random.get_state() 

            torch.manual_seed(config["torch_seed"] + noise_seed_offset)
            np.random.seed(config["np_seed"] + noise_seed_offset)  

            noise = torch.randn_like(grad)/num_grads
            grad = grad + noise
            noisy_analysis_grads.append(grad)
            
            torch.set_rng_state(current_rng_state)
            np.random.set_state(current_np_state)
    """
    ### 消除 ###
    
    # Set up computational graph
    output = model.forward(data)
    loss = 1/2 * loss_function(output, label)
    
    ### grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads])).item()

    #grad_norm = torch.norm(torch.cat([g.view(-1) for g in noisy_analysis_grads ])).item()

    ### gradient_norms[step] = grad_norm
    ### update_matrix = torch.cat([(-0.01 * g).view(-1) for g in grads])

    #update_matrix = torch.cat([(-0.01 * g).view(-1) for g in noisy_analysis_grads ])

    ### update_norm = torch.norm(update_matrix).item()
    ### update_matrix_norms[step] = update_norm
    
    ### eigenvalues, top_eigenvectors = compute_hessian_eigen(loss, model.parameters())
    
    # eigenvalues, top_eigenvectors = compute_hessian_eigen_pyhessian(model.parameters(), loss, )
    ### hessian_eigenvalues[step] = eigenvalues
    ### dominant_space = compute_dominant_projection_matrix(top_eigenvectors, k = 5)
    # 拼接所有参数的梯度
    ###grad_flat = torch.cat([g.view(-1) for g in grads]).to(device)  # 直接使用 grads

    #grad_flat = torch.cat([g.view(-1) for g in noisy_analysis_grads]).to(device)

    ### projection = dominant_space @ grad_flat
    ### dominant_projection[step] = projection.norm().item()
    print(f"Step {step}: Loss = {loss.item()}")
    
    ### invariant_w1, invariant_w2 = compute_invariant_matrix(W1, W2)
    ### invariant_w1_norm = np.linalg.norm(invariant_w1)
    ### invariant_w2_norm = np.linalg.norm(invariant_w2)

    ### invariant_matrix_w1[step]  = invariant_w1_norm
    ### invariant_matrix_w2[step] = invariant_w2_norm

    """
    Hessian_max_eigenvectors = top_eigenvectors[:, 0].reshape(1, -1)
    cos_similarity_Between_Hessian_invariant = abs(cosine_similarity(invariant_eigenvectors, Hessian_max_eigenvectors)[0][0])
    wandb.log({f"cos_similarity_Between_Hessian_invariant{step}": wandb.Histogram(cos_similarity_Between_Hessian_invariant)}, step=step)
    cos_similarity_Hessian_invariant[step] = cos_similarity_Between_Hessian_invariant
    """
    
    ### record_top_eigenvectors[step] = top_eigenvectors
    ### record_eigenvalues[step] = eigenvalues
        
    ### if step in selected_steps:
        ### wandb.log({f"hessian_eigenvalues_step_{step}": wandb.Histogram(eigenvalues)}, step=step)

        
        
        ### wandb.log({f"dominant_projection_norm_step_{step}": dominant_projection[step]}, step=step)

        
        ### wandb.log({f"gradient_norm_step_{step}": gradient_norms[step]}, step=step)

        
        ### wandb.log({f"update_matrix_norm_step_{step}": update_matrix_norms[step]}, step=step)

# 在训练结束后调用绘图函数
# Selected steps
# plotting.plot_loss_curve(loss_history)
# plotting.plot_hessian_eigenvalues(hessian_eigenvalues)
# plotting.plot_projection_norm(dominant_projection)
# plotting.plot_gradient_norms(gradient_norms)
# plotting.plot_update_matrix_norms(update_matrix_norms)
# plotting.plot_invariant_matrix_norms(invariant_matrix_w1)
# plotting.plot_invariant_matrix_norms(invariant_matrix_w2) 

# Save records
""" change
if not os.path.exists("/home/ouyangzl/BaseLine/2 NN/records"):
    os.makedirs("/home/ouyangzl/BaseLine/2 NN/records")
"""

# Save record_gradients

with open('/home/ouyangzl/BaseLine/2 NN/records/record_gradients_Adam_full.pkl', 'wb') as f:
    pickle.dump(record_grads, f)


# Save eigenvalues
""" change
with open('/home/ouyangzl/BaseLine/2 NN/records/record_eigenvalues_SAM_full_0.98.pkl', 'wb') as f:
    pickle.dump(record_eigenvalues, f)
"""

# Save record_top_eigenvectors
"""
with open('/home/ouyangzl/BaseLine/2 NN/records/record_top_eigenvectors_Adam_full_0.98.pkl', 'wb') as f:
    pickle.dump(record_top_eigenvectors, f)
"""

# with open('/home/ouyangzl/BaseLine/2 NN/records/recorded_steps_top_eigenvectors_SGD_0.98.pkl', 'rb') as f:
    # recorded_steps_top_eigenvectors = pickle.load(f)

"""  change
with open('/home/ouyangzl/BaseLine/2 NN/records/record_top_eigenvectors_SGD_full_0.98.pkl', 'rb') as f:
    record_top_eigenvectors = pickle.load(f)

print(len(record_top_eigenvectors))
"""  


# top_k_trajectory = Top_Down(record_top_eigenvectors, tau, k_0 = None)

# with open('/home/ouyangzl/BaseLine/2 NN/records/top_k_trajectory_Adam_full_0.98.pkl', 'wb') as f:
#    pickle.dump(top_k_trajectory, f)
    
"""  change
with open('/home/ouyangzl/BaseLine/2 NN/records/top_k_trajectory_SGD_full_0.98.pkl', 'rb') as f:
    top_k_trajectory = pickle.load(f)

print(len(top_k_trajectory))
"""  
""" change
with open('/home/ouyangzl/BaseLine/2 NN/records/top_k_trajectory_SGD_full_0.98.pkl', 'rb') as f:
    top_k_trajectory = pickle.load(f)
"""

# Selet steps
# selected_top_eigenvectors = {step: record_top_eigenvectors[step] for step in selected_steps}

# selected_top_k_trajectory = {step: top_k_trajectory[step] for step in selected_steps} 

# Full steps
"""
successive_pca_spectrum = Successive_Record_Steps_PCA(record_top_eigenvectors)
first_last_pca_spectrum = First_Last_Record_Steps_PCA(record_top_eigenvectors)
successive_cos_similarity = Successive_Record_Steps_COS_Similarity(record_top_eigenvectors)
first_last_cos_similarity = First_Last_Record_Steps_COS_Similarity(record_top_eigenvectors)
successive_check_dominant_space = Successive_Check_Dominant_Space(record_top_eigenvectors)
first_last_check_dominant_space = First_Last_Check_Dominant_Space(record_top_eigenvectors)
"""


""""
successive_invariant_cos_similarity = Successive_Record_Steps_COS_Similarity(recorded_steps_invariant_eigenvectors)
first_last_invariant_cos_similarity = First_Last_Record_Steps_COS_Similarity(recorded_steps_invariant_eigenvectors)
"""

"""
first_max_eigenvector = record_top_eigenvectors[record_steps[0]]
last_record_step = record_steps[-1]
last_max_eigenvector = record_top_eigenvectors[last_record_step]
final_eigenvector_norm_diff = np.linalg.norm(last_max_eigenvector - first_max_eigenvector)
"""

# first_max_eigenvector = selected_top_eigenvectors[selected_steps[0]]
# last_record_step = selected_steps[-1]
# last_max_eigenvector = selected_top_eigenvectors[last_record_step]
# final_eigenvector_norm_diff = np.linalg.norm(last_max_eigenvector - first_max_eigenvector)
# wandb.log({"final_eigenvector_norm_difference": final_eigenvector_norm_diff})



# Use vectors
# successive_pca_spectrum = Successive_Record_Steps_PCA(selected_top_eigenvectors)
# first_last_pca_spectrum = First_Last_Record_Steps_PCA(selected_top_eigenvectors)
# successive_cos_similarity = Successive_Record_Steps_COS_Similarity(selected_top_eigenvectors)
# first_last_cos_similarity = First_Last_Record_Steps_COS_Similarity(selected_top_eigenvectors)
# successive_check_dominant_space = Successive_Check_Dominant_Space(selected_top_eigenvectors)
#first_last_check_dominant_space = First_Last_Check_Dominant_Space(selected_top_eigenvectors)

# Use vectors
# plotting.plot_cosine_similarity(successive_cos_similarity) # finished
# plotting.plot_pca_spectrum(successive_pca_spectrum) # finished
# plotting.plot_cosine_similarity_to_last(first_last_cos_similarity) # finished 
# plotting.plot_pca_top_k_eigenvectors(first_last_pca_spectrum)
# plotting.plot_successive_check(successive_check_dominant_space) # finished
# plotting.plot_first_last_check(first_last_check_dominant_space) # finished
# plotting.plot_top_k_trajectory(selected_top_k_trajectory, tau) # finished

"""
plotting.plot_invariant_cosine_similarity(successive_invariant_cos_similarity)
plotting.plot_cosine_similarity_to_last(first_last_invariant_cos_similarity)
plotting.plot_Hessain_invariant_cosine_similarity(cos_similarity_Hessian_invariant)
"""

# 12. 完成 wandb 运行
wandb.finish()