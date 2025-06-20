import numpy as np
import wandb
from sklearn.preprocessing import normalize 
import copy

def normalize_eigenvectors_by_row(recorded_steps_top_eigenvectors):
    """
    Normalize each eigenvector (row) within each matrix in the input dictionary.

    Args:
        recorded_steps_top_eigenvectors (dict): A dictionary where keys are steps 
                                                and values are matrices (e.g., NumPy arrays)
                                                of shape (N, D), where N is the number 
                                                of top eigenvectors and D is the dimension.
                                                Each row represents an eigenvector.

    Returns:
        dict: A new dictionary with the same keys, but where each value is a matrix
              containing the row-normalized eigenvectors (L2 norm by default).
    """
    normalized_dict = {}
    
    for step, eigenvector_matrix in recorded_steps_top_eigenvectors.items():
        # Ensure the input is array-like (e.g., numpy array or can be converted)
        # Using copy() can prevent modification of the original data if it's mutable
        # although normalize usually returns a new array. Being explicit can be safer.
        # Convert to numpy array just in case it's not, copy to be safe
        matrix_to_normalize = eigenvector_matrix.detach().cpu().numpy()

        normalized_matrix = normalize(matrix_to_normalize, axis=0, norm='l2')
        
        normalized_dict[step] = normalized_matrix
        
    return normalized_dict

def Successive_Record_Steps_PCA(recorded_steps_top_eigenvectors):
    recorded_steps_top_eigenvectors = normalize_eigenvectors_by_row(recorded_steps_top_eigenvectors)
    Successive_Record_Steps_PCA_Spectrum = {}
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
    num_steps = len(sorted_steps)

    for i in range(num_steps):
        current_step = sorted_steps[i]
        current_space = recorded_steps_top_eigenvectors[current_step]
        if i > 0:
            previous_step = sorted_steps[i - 1]
            previous_space = recorded_steps_top_eigenvectors[previous_step]
            
            # combined_vectors = np.concatenate((np.real(current), np.real(previous)), axis=1)
            # eigenvalues = np.linalg.eigvalsh(matrix)
            
            matrix = np.matmul(current_space.T, previous_space)
            _, sigma, _ = np.linalg.svd(matrix)
            
            sorted_eigenvalues = np.sort(sigma)[::-1]
            Successive_Record_Steps_PCA_Spectrum[current_step] = sorted_eigenvalues
            wandb.log({f"successive_pca_spectrum_step_{current_step}": wandb.Histogram(sorted_eigenvalues)}, step=current_step)

    return Successive_Record_Steps_PCA_Spectrum

def First_Last_Record_Steps_PCA(recorded_steps_top_eigenvectors):
    recorded_steps_top_eigenvectors = normalize_eigenvectors_by_row(recorded_steps_top_eigenvectors)
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
        wandb.log({f"first_last_pca_spectrum_step_{current_step}": wandb.Histogram(sorted_eigenvalues)}, step=current_step)

    return First_Last_Record_Steps_PCA_Spectrum