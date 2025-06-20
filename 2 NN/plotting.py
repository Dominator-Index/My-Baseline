import numpy as np
import matplotlib.pyplot as plt
import wandb
from config import config

import os

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

def plot_top_eigenvalue():
    and
    
def plot_hessian_eigenvalues(hessian_eigenvalues):
    plt.figure(figsize=(12, 8))
    # num_subplots = len(config["record_steps"])
    selected_steps = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
    num_subplots = len(selected_steps)
    rows = (num_subplots + 4) // 5  # 向上取整计算行数
    cols = min(num_subplots, 5)
    for i, step in enumerate(selected_steps):  # config["record_steps"]
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
    plt.figure(figsize=(18, 10))

    rows = (len(pca_spectrum) + 4) // 5
    cols = min(len(pca_spectrum), 5)

    for i, step in enumerate(pca_spectrum.keys()):  # Iterate over the steps in pca_spectrum
        plt.subplot(rows, cols, i + 1)
        spectrum = pca_spectrum[step]
        
        # Plot the sorted eigenvalues
        plt.plot(range(1, len(spectrum) + 1), sorted(spectrum, reverse=True), marker='o', linestyle='-', color='b')

        plt.title(f"Step {step} PCA Spectrum", fontsize=8)
        plt.xlabel("Index", fontsize=8)
        plt.ylabel("Eigenvalue",fontsize=8)

        # Set y-axis limits close to 1
        # plt.ylim(0.99, 1.01)  # Adjust this range based on the actual distribution of your data
    
    # Set the overall title
    plt.suptitle("PCA Spectrum of Top-k Largest Hessian Eigenvectors at Adjacent Recorded Steps")

    # Adjust the layout to prevent overlap
    plt.subplots_adjust(hspace=1.5, wspace=0.6)  # Adjust vertical and horizontal space between subplots

    # Save the figure as a PNG file
    plt.savefig(os.path.join('/home/ouyangzl/BaseLine/2 NN', "PCA_Spectrum_of_Top_k_Largest_Hessian_Eigenvectors.png"))
    
    wandb.log({"PCA Spectrum of Top-k Largest Hessian Eigenvectors at Adjacent Recorded Steps (SAM)": wandb.Image(plt)})

    # Show the plot
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

def plot_cosine_similarity_to_last(first_last_cos_similarity):
    if first_last_cos_similarity:
        steps, similarities = zip(*first_last_cos_similarity)
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

def plot_successive_check(Successive_Check_Dominant_Space):
    """
    绘制 Successive_Check_Dominant_Space 函数结果的折线图。

    Args:
        Successive_Check_Dominant_Space (dict): Successive_Check_Dominant_Space 函数返回的字典，
                     键是步骤数，值是投影范数之差。
    """
    if not Successive_Check_Dominant_Space:
        print("No data to plot for Successive Check.")
        return

    steps = sorted(Successive_Check_Dominant_Space.keys())
    values = [Successive_Check_Dominant_Space[step] for step in steps]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Projection Norm Difference ")
    plt.title("Difference Norm of The Largest Eigenvector and Its Projection on Dominant Space at Adjacent Recorded Steps")
    plt.grid(True)
    wandb.log({"Difference Norm of The Largest Eigenvector and Its Projection on Dominant Space at Adjacent Recorded Steps": wandb.Image(plt)})
    plt.show()

def plot_first_last_check(First_Last_Check_Dominant_Space):
    """
    绘制 First_Last_Check_Dominant_Space 函数结果的折线图。

    Args:
        First_Last_Check_Dominant_Space (dict): First_Last_Check_Dominant_Space 函数返回的字典，
                     键是步骤数，值是投影范数之差。
    """
    if not First_Last_Check_Dominant_Space:
        print("No data to plot for First Last Check.")
        return

    steps = sorted(First_Last_Check_Dominant_Space.keys())
    values = [First_Last_Check_Dominant_Space[step] for step in steps]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, values, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Projection Norm Difference ")
    plt.title("Difference Norm of The Largest Eigenvector and Its Projection on Dominant Space Between Different Recorded Steps and The Last Recorded Steps")
    plt.grid(True)
    wandb.log({"Difference Norm of The Largest Eigenvector and Its Projection on Dominant Space Between Different Recorded Steps and The Last Recorded Steps": wandb.Image(plt)})
    plt.show()


def plot_invariant_cosine_similarity(successive_invariant_cos_similarity):
    if successive_invariant_cos_similarity:
        steps, similarities = zip(*successive_invariant_cos_similarity)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, similarities, marker='o', linestyle='-', color='red')
        plt.xlabel("Step")
        plt.ylabel("Cosine Similarity")
        plt.title("Cosine Similarity of The Invariant Eigenvectors at Adjacent Recorded Steps")
        plt.grid(True)
        wandb.log({"Cosine Similarity of The Invariant Hessian Eigenvectors at Adjacent Recorded Steps": wandb.Image(plt)})
        plt.show()
    else:
        print("No cosine similarities to plot.")


def plot_cosine_similarity_to_last(first_last_invariant_cos_similarity ):
    if first_last_invariant_cos_similarity :
        steps, similarities = zip(*first_last_invariant_cos_similarity )
        plt.figure(figsize=(8, 5))
        plt.plot(steps, similarities, marker='o', linestyle='-', color='red')
        plt.xlabel("Step")
        plt.ylabel("Cosine Similarity")
        plt.title("Cosine Similarity of The Invariant Eigenvectors Between Different Recorded Steps and The Last Recorded Steps")
        plt.grid(True)
        wandb.log({"Cosine Similarity of The Invariant Eigenvectors Between Different Recorded Steps and The Last Recorded Steps": wandb.Image(plt)})
        plt.show()
    else:
        print("No cosine similarities to plot.")


def plot_Hessain_invariant_cosine_similarity(cos_similarity_Hessian_invariant):
    if cos_similarity_Hessian_invariant:
        steps, similarities = zip(*cos_similarity_Hessian_invariant)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, similarities, marker='o', linestyle='-', color='red')
        plt.xlabel("Step")
        plt.ylabel("Cosine Similarity")
        plt.title("Cosine Similarity of The Invariant Eigenvectors and The Largest Hessian Eigenvectors at Adjacent Recorded Steps")
        plt.grid(True)
        wandb.log({"Cosine Similarity of The Invariant Eigenvectors and The Largest Hessian Eigenvectors at Adjacent Recorded Steps": wandb.Image(plt)})
        plt.show()
    else:
        print("No cosine similarities to plot.")

def plot_invariant_matrix_norms(invariant_matrix):
    plt.figure(figsize=(8, 5))
    steps_proj = sorted(invariant_matrix.keys())  # 按步骤排序
    norms_proj = [invariant_matrix[step] for step in steps_proj]
    plt.plot(steps_proj, norms_proj, marker='o', linestyle='-', color='green')
    plt.xlabel("Step")
    plt.ylabel("Invariant matrix norms")
    plt.title("Invariant matrix Norm at Different Recorded Steps")
    plt.grid(True)
    wandb.log({"Invariant matrix at Different Recorded Steps": wandb.Image(plt)})
    plt.show()
    
def plot_top_k_trajectory(trajectory, tau):
    plt.figure(figsize=(16, 6))  # 设置画布尺寸（示例：12x6英寸）
    length = len(trajectory)
    x = np.arange(length)  # 生成横轴 0, 1, 2, ..., steps-1
    trajectory_list = list(trajectory.values())
    plt.plot(x, trajectory_list, marker='o', linestyle='-', linewidth=2, markersize=6)  # 画折线图
    # 添加图表标注
    plt.title("Top-k Trajectory with Tolerance {}".format(tau), fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Top-k", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)  # 显示网格线
    """
    # 设置自定义的y轴刻度
    # 前半部分刻度：以10为单位增长
    y_ticks_first = np.arange(0, 100, 10)
    # 后半部分刻度：以400为单位增长
    y_ticks_second = np.arange(100, 450, 50)
    # 合并前后部分的刻度
    y_ticks = np.concatenate((y_ticks_first, y_ticks_second))
    plt.yticks(y_ticks)  # 设置y轴刻度
    """
    wandb.log({"Top k Trajectory at Recorded Steps with Tolerance {}".format(tau): wandb.Image(plt)})
    plt.tight_layout()  # 自动调整子图间距
    plt.savefig("/home/ouyangzl/BaseLine/2 NN/images/Top k Trajectory at Recorded Steps with Tolerance {}.png".format(tau))
    plt.show()