import numpy as np
import wandb

def Successive_Record_Steps_PCA(recorded_steps_top_eigenvectors):
    Successive_Record_Steps_PCA_Spectrum = {}
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
    num_steps = len(sorted_steps)

    for i in range(num_steps):
        current_step = sorted_steps[i]
        current_space = recorded_steps_top_eigenvectors[current_step][:, :100]
        if i > 0:
            previous_step = sorted_steps[i - 1]
            previous_space = recorded_steps_top_eigenvectors[previous_step][:, :100]
            
            # combined_vectors = np.concatenate((np.real(current), np.real(previous)), axis=1)
            # eigenvalues = np.linalg.eigvalsh(matrix)
            
            matrix = np.matmul(current_space.T, previous_space)
            _, sigma, _ = np.linalg.svd(matrix)
            
            sorted_eigenvalues = np.sort(sigma)[::-1][:40]
            Successive_Record_Steps_PCA_Spectrum[current_step] = sorted_eigenvalues[:40]
            # wandb.log({f"successive_pca_spectrum_step_{current_step}": wandb.Histogram(sorted_eigenvalues[:40], )}, step=current_step)

    return Successive_Record_Steps_PCA_Spectrum

def First_Last_Record_Steps_PCA(recorded_steps_top_eigenvectors):
    First_Last_Record_Steps_PCA_Spectrum = {}
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
    num_steps = len(sorted_steps)
    if num_steps < 2:
        return First_Last_Record_Steps_PCA_Spectrum  # Return empty if less than two steps recorded

    last_step = sorted_steps[-1]
    Last_top_eigenvectors = recorded_steps_top_eigenvectors[last_step]

    for i in range(num_steps - 1):
        current_step = sorted_steps[i]
        current = recorded_steps_top_eigenvectors[current_step]
        # combined_vectors = np.concatenate((np.real(current), np.real(Last_top_eigenvectors)), axis=1)
        # matrix = np.matmul(combined_vectors.T, combined_vectors)
        # eigenvalues = np.linalg.eigvalsh(matrix)
        
        matrix = np.matmul(current.T, Last_top_eigenvectors)
        _, sigma, _ = np.linalg.svd(matrix)
        sorted_eigenvalues = np.sort(sigma)[::-1]
        First_Last_Record_Steps_PCA_Spectrum[current_step] = sorted_eigenvalues
        # wandb.log({f"first_last_pca_spectrum_step_{current_step}": wandb.Histogram(sorted_eigenvalues)}, step=current_step)

    return First_Last_Record_Steps_PCA_Spectrum