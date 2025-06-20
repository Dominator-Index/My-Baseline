import matplotlib.pyplot as plt
import numpy as np
import wandb
from config import config

def plot_loss_curve(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loss_history)), loss_history, label="Training Loss", color="blue")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    wandb.log({"loss_curve": wandb.Image(plt)})
    plt.show()

def plot_hessian_eigenvalues(hessian_eigenvalues):
    plt.figure(figsize=(12, 8))
    num_subplots = len(config["record_steps"])
    rows = (num_subplots + 4) // 5  # 向上取整计算行数
    cols = min(num_subplots, 5)
    for i, step in enumerate(config["record_steps"]):
        plt.subplot(rows, cols, i + 1)
        eigenvalues = hessian_eigenvalues[step]
        if eigenvalues.size > 0:
            plt.hist(eigenvalues, bins=50, color='blue', alpha=0.7)
        plt.title(f"Step {step}")
        plt.xlabel("Eigenvalues")
        plt.ylabel("Frequency")
   
    plt.suptitle("Empirical Spectral Distribution at Different Recorded Steps")  # 添加总标题
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整子图布局，为总标题留出空间
    
    wandb.log({"Empirical Spectral Distribution at Different Recorded Steps": wandb.Image(plt)})
    plt.show()

def plot_cosine_similarity(successive_cos_similarity):
    if successive_cos_similarity:
        steps, similarities = zip(*successive_cos_similarity)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, similarities, marker='o', linestyle='-', color='red')
        plt.xlabel("Step")
        plt.ylabel("Cosine Similarity")
        plt.title("Cosine Similarity of The Largest Hessian Eigenvectors at Adjacent Recorded Steps")
        plt.grid(True)
        wandb.log({"Cosine Similarity of The Largest Hessian Eigenvectors at Adjacent Recorded Steps": wandb.Image(plt)})
        plt.show()
    else:
        print("No cosine similarities to plot.")

def plot_pca_spectrum(pca_spectrum):
    plt.figure(figsize=(12, 8))
    record_steps_plotting = config["record_steps"][1:]
    num_subplots = len(record_steps_plotting)
    rows = (num_subplots + 4) // 5
    cols = min(num_subplots, 5)
    for i, step in enumerate(record_steps_plotting):
        plt.subplot(rows, cols, i + 1)
        spectrum = pca_spectrum[step]
        plt.plot(range(1, len(spectrum) + 1), sorted(spectrum, reverse=True), marker='o', linestyle='-')
        plt.title(f"Step {step} PCA Spectrum")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
    plt.suptitle("PCA Spectrum of Top-k Largest Hessian Eigenvectors at Adjacent Recorded Steps")  # 添加主标题
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整子图布局，为总标题留出空间
    wandb.log({"PCA Spectrum of Top-k Largest Hessian Eigenvectors at Adjacent Recorded Steps": wandb.Image(plt)})
    plt.show()

def plot_projection_norm(dominant_projection):
    plt.figure(figsize=(8, 5))
    steps_proj = sorted(dominant_projection.keys())  # 按步骤排序
    norms_proj = [dominant_projection[step] for step in steps_proj]
    plt.plot(steps_proj, norms_proj, marker='o', linestyle='-', color='green')
    plt.xlabel("Step")
    plt.ylabel("Projection Norm")
    plt.title("Gradient Norm in Dominant Subspace at Different Recorded Steps")
    plt.grid(True)
    wandb.log({"Gradient Norm in Dominant Subspace at Different Recorded Steps": wandb.Image(plt)})
    plt.show()

def plot_gradient_norms(gradient_norms):
    plt.figure(figsize=(8, 5))
    steps_proj = sorted(gradient_norms.keys())  # 按步骤排序
    norms_proj = [gradient_norms[step] for step in steps_proj]
    plt.plot(steps_proj, norms_proj, marker='o', linestyle='-', color='green')
    plt.xlabel("Step")
    plt.ylabel("Gradient norms")
    plt.title("Gradient Norm at Different Recorded Steps")
    plt.grid(True)
    wandb.log({"Gradient Norm at Different Recorded Steps": wandb.Image(plt)})
    plt.show()

def plot_update_matrix_norms(update_matrix_norms):
    plt.figure(figsize=(8, 5))
    steps_proj = sorted(update_matrix_norms.keys())  # 按步骤排序
    norms_proj = [update_matrix_norms[step] for step in steps_proj]
    plt.plot(steps_proj, norms_proj, marker='o', linestyle='-', color='green')
    plt.xlabel("Step")
    plt.ylabel("Update matrix norms")
    plt.title("Norm of Update Matrix at Different Recorded Steps")
    plt.grid(True)
    wandb.log({"Norm of Update Matrix at Different Recorded Steps": wandb.Image(plt)})
    plt.show()

def plot_cosine_similarity_to_last(first_last_pca_similarity):
    if first_last_pca_similarity:
        steps, similarities = zip(*first_last_pca_similarity)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, similarities, marker='o', linestyle='-', color='red')
        plt.xlabel("Step")
        plt.ylabel("Cosine Similarity")
        plt.title("Cosine Similarity of The Largest Hessian Eigenvectors Between Different Recorded Steps and The Last Recorded Steps")
        plt.grid(True)
        wandb.log({"Cosine Similarity of The Largest Hessian Eigenvectors Between Different Recorded Steps and The Last Recorded Steps": wandb.Image(plt)})
        plt.show()
    else:
        print("No cosine similarities to plot.")

# 在 plotting.py 中修改 plot_pca_top_k_eigenvectors 函数
def plot_pca_top_k_eigenvectors(first_last_pca_spectrum):
    plt.figure(figsize=(12, 8))
    record_steps_plotting = config["record_steps"][1:]
    steps_to_plot = [step for step in record_steps_plotting if step in first_last_pca_spectrum] # 只绘制 first_last_pca_spectrum 中存在的步骤
    num_subplots = len(steps_to_plot)
    rows = (num_subplots + 4) // 5
    cols = min(num_subplots, 5)
    for i, step in enumerate(steps_to_plot):
        plt.subplot(rows, cols, i + 1)
        spectrum = first_last_pca_spectrum[step]
        plt.plot(range(1, len(spectrum) + 1), sorted(spectrum, reverse=True), marker='o', linestyle='-')
        plt.title(f"Step {step} PCA Spectrum")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
    plt.suptitle("PCA Spectrum of Top-k Largest Hessian Eigenvectors Between Different Recorded Steps and The Last Recorded Steps")  # 添加主标题
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整子图布局，为总标题留出空间
    wandb.log({"PCA Spectrum of Top-k Largest Hessian Eigenvectors Between Different Recorded Steps and The Last Recorded Steps": wandb.Image(plt)})
    plt.show()

