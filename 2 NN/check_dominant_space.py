import torch
from config import config, device  
import wandb
from hessian_utils import compute_dominant_projection_matrix

def Successive_Check_Dominant_Space(recorded_steps_top_eigenvectors):
    Successive_Check_Dominant_Space_results = {}
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
    num_steps = len(sorted_steps)

    for i in range(num_steps):
        current_step = sorted_steps[i]
        current_numpy = recorded_steps_top_eigenvectors[current_step]

        current_space = compute_dominant_projection_matrix(current_numpy, config["top_k_pca_number"])
        if i > 0:
            previous_step = sorted_steps[i - 1]
            previous_numpy = recorded_steps_top_eigenvectors[previous_step][:, 0].reshape(1, -1)
            previous = torch.tensor(previous_numpy).float().to(device)
            previous_norm = torch.norm(previous).to(device)
            Projection = torch.matmul(current_space, previous.T).to(device)
            Projection_norm = torch.norm(Projection).to(device)
            Successive_Check_Dominant_Space_results[current_step] = (Projection_norm - previous_norm).item()
            wandb.log({f" Successive_Check_Dominant_Space{current_step}": wandb.Histogram((Projection_norm - previous_norm).cpu().numpy())}, step=current_step)

    return  Successive_Check_Dominant_Space_results

def First_Last_Check_Dominant_Space(recorded_steps_top_eigenvectors):
    First_Last_Check_Dominant_Space_results = {}
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
    num_steps = len(sorted_steps)
    if num_steps < 2:
        return First_Last_Check_Dominant_Space_results  # Return empty if less than two steps recorded

    last_step = sorted_steps[-1]
    Last_top_eigenvectors_numpy = recorded_steps_top_eigenvectors[last_step]

    Last_dominant_space = compute_dominant_projection_matrix(Last_top_eigenvectors_numpy, config["top_k_pca_number"])

    for i in range(num_steps - 1):
        current_step = sorted_steps[i]
        current_numpy = recorded_steps_top_eigenvectors[current_step][:, 0].reshape(1, -1)
        current = torch.tensor(current_numpy).float().to(device)
        current_norm = torch.norm(current).to(device)
        Projection = torch.matmul(Last_dominant_space, current.T).to(device)
        Projection_norm = torch.norm(Projection).to(device)
        First_Last_Check_Dominant_Space_results[current_step] = (Projection_norm - current_norm).item()
        wandb.log({f"First_Last_Check_Dominant_Space{current_step}": wandb.Histogram((Projection_norm - current_norm).cpu().numpy())}, step=current_step)

    return First_Last_Check_Dominant_Space_results