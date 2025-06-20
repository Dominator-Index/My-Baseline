import numpy as np
from sklearn.decomposition import PCA
from config import config
import wandb

def Successive_Record_Steps_PCA(recorded_steps_top_eigenvectors):
    Successive_Record_Steps_PCA_Spectrum = {}
    sorted_steps = sorted(recorded_steps_top_eigenvectors.keys())
    num_steps = len(sorted_steps)

    for i in range(num_steps):
        current_step = sorted_steps[i]
        current = recorded_steps_top_eigenvectors[current_step]
        if i > 0:
            previous_step = sorted_steps[i - 1]
            previous = recorded_steps_top_eigenvectors[previous_step]
            combined_vectors = np.concatenate((np.real(current), np.real(previous)), axis=1)
            pca = PCA()
            pca.fit(combined_vectors.T)
            Successive_Record_Steps_PCA_Spectrum[current_step] = pca.explained_variance_
            wandb.log({f"successive_pca_spectrum_step_{current_step}": wandb.Histogram(pca.explained_variance_)}, step=current_step)

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
        combined_vectors = np.concatenate((np.real(current), np.real(Last_top_eigenvectors)), axis=1)
        pca = PCA()
        pca.fit(combined_vectors.T)
        First_Last_Record_Steps_PCA_Spectrum[current_step] = pca.explained_variance_
        wandb.log({f"first_last_pca_spectrum_step_{current_step}": wandb.Histogram(pca.explained_variance_)}, step=current_step)

    return First_Last_Record_Steps_PCA_Spectrum