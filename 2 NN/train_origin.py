import torch
import torch.optim as optim
import numpy as np
import wandb
from config import config, device


from model import LinearNetwork
from data_utils import generate_low_rank_identity
from loss_utils import loss_fn
from hessian_utils import compute_hessian_eigen, compute_dominant_projection_matrix

import plotting
from PCA import Successive_Record_Steps_PCA, First_Last_Record_Steps_PCA
from COS_similarity import Successive_Record_Steps_COS_Similarity, First_Last_Record_Steps_COS_Similarity
from check_dominant_space import Successive_Check_Dominant_Space, First_Last_Check_Dominant_Space
from Compute_invariant_eigenvector import compute_invariant_matrix

# 设置随机种子
np.random.seed(config["np_seed"])
torch.manual_seed(config["torch_seed"])

# 初始化 wandb
wandb.init(project=config["wandb_project_name"], name=config["wandb_run_name"])

# 使用 config 字典更新 wandb.config
wandb.config.update(config)

# 初始化模型和优化器
model = LinearNetwork(config["input_dim"], config["hidden_dim"], config["output_dim"], device).to(device)
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

# 训练过程
steps = config["steps"]
record_steps = config["record_steps"]
hessian_eigenvalues = {}
loss_history = []
dominant_projection = {}
gradient_norms = {}
update_matrix_norms = {}
recorded_steps_top_eigenvectors = {} 

recorded_steps_invariant_marix_w1 = {}
recorded_steps_invariant_marix_w2 = {}


for step in range(steps + 1):
    optimizer.zero_grad()
    output = model.forward()
    target = generate_low_rank_identity(config["input_dim"], config["output_dim"])
    loss = loss_fn(output, target)
    loss_history.append(loss.item())

    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    
    """
    compute the number of grads and set different seed to generative seed
   
    num_grads = sum(1 for g in grads if g is not None)
    noise_seed_offset = step  
    """

    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            
            """
            current_rng_state = torch.get_rng_state()
            current_np_state = np.random.get_state() 

            torch.manual_seed(config["torch_seed"] + noise_seed_offset)
            np.random.seed(config["np_seed"] + noise_seed_offset) 

            noise = torch.randn_like(grad)/num_grads
            grad = grad + noise
            """

            param.data.copy_(param - 0.01 * grad)  # 避免 in-place 修改

            """
            torch.set_rng_state(current_rng_state)
            np.random.set_state(current_np_state)
            """

    wandb.log({"loss": loss.item()}, step=step)

    if step in record_steps:
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

        output = model.forward()
        loss = loss_fn(output, target)
        grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads])).item()

        #grad_norm = torch.norm(torch.cat([g.view(-1) for g in noisy_analysis_grads ])).item()

        gradient_norms[step] = grad_norm
        update_matrix = torch.cat([(-0.01 * g).view(-1) for g in grads])

        #update_matrix = torch.cat([(-0.01 * g).view(-1) for g in noisy_analysis_grads ])
    
        update_norm = torch.norm(update_matrix).item()
        update_matrix_norms[step] = update_norm
        eigenvalues, top_eigenvectors = compute_hessian_eigen(loss, model.parameters())
        hessian_eigenvalues[step] = eigenvalues
        dominant_space = compute_dominant_projection_matrix(top_eigenvectors, k = 5)
        # 拼接所有参数的梯度
        grad_flat = torch.cat([g.view(-1) for g in grads]).to(device)  # 直接使用 grads

        #grad_flat = torch.cat([g.view(-1) for g in noisy_analysis_grads]).to(device)

        projection = dominant_space @ grad_flat
        dominant_projection[step] = projection.norm().item()
        print(f"Step {step}: Loss = {loss.item()}")
        
        invariant_w1, invariant_w2 = compute_invariant_matrix(W1, W2)
        invariant_w1_norm = np.linalg.norm(invariant_w1)
        invariant_w2_norm = np.linalg.norm(invariant_w2)

        recorded_steps_invariant_marix_w1[step] = invariant_w1_norm
        recorded_steps_invariant_marix_w2[step] = invariant_w2_norm

        """
        Hessian_max_eigenvectors = top_eigenvectors[:, 0].reshape(1, -1)
        cos_similarity_Between_Hessian_invariant = abs(cosine_similarity(invariant_eigenvectors, Hessian_max_eigenvectors)[0][0])
        wandb.log({f"cos_similarity_Between_Hessian_invariant{step}": wandb.Histogram(cos_similarity_Between_Hessian_invariant)}, step=step)
        cos_similarity_Hessian_invariant[step] = cos_similarity_Between_Hessian_invariant
        """

        wandb.log({f"hessian_eigenvalues_step_{step}": wandb.Histogram(eigenvalues)}, step=step)

        recorded_steps_top_eigenvectors[step] = top_eigenvectors
    
       
        wandb.log({f"dominant_projection_norm_step_{step}": dominant_projection[step]}, step=step)

        
        wandb.log({f"gradient_norm_step_{step}": gradient_norms[step]}, step=step)

        
        wandb.log({f"update_matrix_norm_step_{step}": update_matrix_norms[step]}, step=step)


successive_pca_spectrum = Successive_Record_Steps_PCA(recorded_steps_top_eigenvectors)
first_last_pca_spectrum = First_Last_Record_Steps_PCA(recorded_steps_top_eigenvectors)
successive_cos_similarity = Successive_Record_Steps_COS_Similarity(recorded_steps_top_eigenvectors)
first_last_cos_similarity = First_Last_Record_Steps_COS_Similarity(recorded_steps_top_eigenvectors)
successive_check_dominant_space = Successive_Check_Dominant_Space(recorded_steps_top_eigenvectors)
first_last_check_dominant_space = First_Last_Check_Dominant_Space(recorded_steps_top_eigenvectors)

""""
successive_invariant_cos_similarity = Successive_Record_Steps_COS_Similarity(recorded_steps_invariant_eigenvectors)
first_last_invariant_cos_similarity = First_Last_Record_Steps_COS_Similarity(recorded_steps_invariant_eigenvectors)
"""


first_max_eigenvector = recorded_steps_top_eigenvectors[record_steps[0]]
last_record_step = record_steps[-1]
last_max_eigenvector = recorded_steps_top_eigenvectors[last_record_step]
final_eigenvector_norm_diff = np.linalg.norm(last_max_eigenvector - first_max_eigenvector)
wandb.log({"final_eigenvector_norm_difference": final_eigenvector_norm_diff})



# 在训练结束后调用绘图函数
plotting.plot_loss_curve(loss_history)
plotting.plot_hessian_eigenvalues(hessian_eigenvalues)
plotting.plot_cosine_similarity(successive_cos_similarity)
plotting.plot_pca_spectrum(successive_pca_spectrum)
plotting.plot_projection_norm(dominant_projection)
plotting.plot_gradient_norms(gradient_norms)
plotting.plot_update_matrix_norms(update_matrix_norms)
plotting.plot_cosine_similarity_to_last(first_last_cos_similarity)
plotting.plot_pca_top_k_eigenvectors(first_last_pca_spectrum)
plotting.plot_successive_check(successive_check_dominant_space)
plotting.plot_first_last_check(first_last_check_dominant_space)

"""
plotting.plot_invariant_cosine_similarity(successive_invariant_cos_similarity)
plotting.plot_cosine_similarity_to_last(first_last_invariant_cos_similarity)
plotting.plot_Hessain_invariant_cosine_similarity(cos_similarity_Hessian_invariant)
"""

plotting.plot_invariant_matrix_norms(recorded_steps_invariant_marix_w1)
plotting.plot_invariant_matrix_norms(recorded_steps_invariant_marix_w2)


# 12. 完成 wandb 运行
wandb.finish()