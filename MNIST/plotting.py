import matplotlib.pyplot as plt
import wandb
from MNIST_config import config
import numpy as np
import itertools
def plot_loss_curve(loss_history, title="Training Loss Curve"):
    plt.figure(figsize=(12, 8))
    steps = sorted(loss_history.keys())
    losses = [loss_history[step] for step in steps]
    plt.plot(steps, losses, label=title, color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    wandb.log({title: wandb.Image(plt)})  # 使用图表的标题作为键名
    plt.grid(True)
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

def plot_invariant_matrix_norms(invariant_matrix, title):
    plt.figure(figsize=(8, 5))
    steps_proj = sorted(invariant_matrix.keys())  # 按步骤排序
    norms_proj = [invariant_matrix[step] for step in steps_proj]
    plt.plot(steps_proj, norms_proj, marker='o', linestyle='-', color='green')
    plt.xlabel("Step")
    plt.ylabel("Invariant matrix norms")
    plt.title(f"Invariant matrix {title} Norm at Different Recorded Steps")
    plt.grid(True)
    wandb.log({f"Invariant matrix {title} Norm at Different Recorded Steps": wandb.Image(plt)})
    plt.show()

def plot_train_accuracy(train_accuracy_history):
    """Plots the training accuracy over steps."""
    steps = list(train_accuracy_history.keys())
    accuracies = list(train_accuracy_history.values())
    plt.figure(figsize=(8, 2))
    plt.plot(steps, accuracies, marker='o')
    plt.title('Training Accuracy Over Steps')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.grid(True)
    wandb.log({"Training Accuracy Over Steps": wandb.Image(plt)})
    plt.show()

def plot_top_2k_eigenvalues(eigenvalues):
    steps = sorted(eigenvalues.keys())
    plt.figure(figsize=(12, 6))
    num_eigenvalues = len(eigenvalues[steps[0]]) if steps else 0
    for i in range(num_eigenvalues):
        eigenvalue_sequence = [eigenvalues[step][i] for step in steps]
        plt.plot(steps, eigenvalue_sequence, linewidth=1, label=f'Eigenvalue {i+1}') # 增加线宽并添加标签
    plt.title('The Largest 20 Eigenvalues Over Steps')
    plt.xlabel('Step')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.legend() # 显示图例
    wandb.log({"The Largest 20 Eigenvalues Over Steps": wandb.Image(plt)})
    plt.show()

def plot_X_loss(X_loss):
    plt.figure(figsize=(8, 5))
    steps = sorted(X_loss.keys()) 
    losses = [X_loss[step] for step in steps] 
    plt.plot(steps, losses, label="X Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("X Loss Curve")
    plt.legend()
    wandb.log({"Training X Loss Curve": wandb.Image(plt)})
    plt.grid(True)
    plt.show()

def plot_comparison_loss_with_phases(loss_history, convergence_step):
    """
    绘制训练过程中不同阶段的损失曲线。

    Args:
        loss_history (dict): 包含 'SGD', 'Dominant', 'Bulk' 键的字典，
                             每个键对应的值是一个记录了损失历史的字典。
        convergence_step (int): 初始 SGD 收敛的步数。
    """
    plt.figure(figsize=(10, 6))

    # 绘制 convergence 之前的 SGD loss
    sgd_steps_before = sorted([step for step in loss_history['SGD'].keys() if step <= convergence_step and convergence_step != -1])
    sgd_losses_before = [loss_history['SGD'][step] for step in sgd_steps_before]
    if sgd_steps_before:
        plt.plot(sgd_steps_before, sgd_losses_before, label="SGD (Initial)", color="blue")

    # 绘制 convergence 之后的 loss
    if convergence_step != -1:
        sgd_steps_after = sorted([step for step in loss_history['SGD'].keys() if step >= convergence_step])
        sgd_losses_after = [loss_history['SGD'][step] for step in sgd_steps_after]
        if sgd_steps_after:
            plt.plot(sgd_steps_after, sgd_losses_after, label="SGD (Continued)", color="green")

        dominant_steps = sorted([step for step in loss_history['Dominant'].keys() if step >= convergence_step])
        dominant_losses = [loss_history['Dominant'][step] for step in dominant_steps]
        if dominant_steps:
            plt.plot(dominant_steps, dominant_losses, label="Dominant", color="red")

        bulk_steps = sorted([step for step in loss_history['Bulk'].keys() if step >= convergence_step])
        bulk_losses = [loss_history['Bulk'][step] for step in bulk_steps]
        if bulk_steps:
            plt.plot(bulk_steps, bulk_losses, label="Bulk", color="purple")
    elif convergence_step == -1:
        # 如果没有收敛，只绘制 SGD 的 loss
        sgd_steps = sorted(loss_history['SGD'].keys())
        sgd_losses = [loss_history['SGD'][step] for step in sgd_steps]
        plt.plot(sgd_steps, sgd_losses, label="SGD", color="blue")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss Comparison Across Training Phases")
    plt.legend()
    plt.grid(True)
    wandb.log({"Loss Comparison": wandb.Image(plt)})  # 修改了键名
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
    
def plot_single_cosine_similarity(history, interval=50, label=None, save_path='./images'):
    colors = {'batch gradient': 'red', 'full gradient': 'blue', 'gradient difference': 'green'}
    markers = {'batch gradient': 'o', 'full gradient': 's', 'gradient difference': 'v'}
    
    if not history:
        print(f"No data for {label}")
        return
    
    # Check data structure
    """
    if not all(isinstance(i, tuple) and len(i) == 2 for i in history):
        print(f"Invalid format in history for {label}")
        return
    """
    
    
    # 提取所有的 steps 和 similarities
    # 将 dict 转换为 list，并按照 step 排序（保险起见）
    steps = sorted(history.keys())
    similarities = [history[step] for step in steps]
    
    
    # 筛选出符合 interval 间隔的 steps 和 similarities
    filtered_steps = [step for step in steps if step % interval == 0]
    filtered_similarities = [history[step] for step in filtered_steps]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # 绘制余弦相似度曲线
    plt.plot(filtered_steps, filtered_similarities,
             marker=markers.get(label, 'o'),
             linestyle='-',
             color=colors.get(label, 'black'),
             label=label)
    
    # 添加标签和标题
    plt.xlabel("Step")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Cosine similarity between {label} and Hessian's top eigenvector")
    plt.grid(True)
    plt.legend()
    
    # 上传图像到 wandb
    wandb.log({f"Cosine similarity between {label} and Hessian's top eigenvector": wandb.Image(plt)})
    
    # 保存图像到指定路径
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # 显示图像
    plt.show()

def plot_cosine_similarity_all(histories, interval=50, save_path='./images'):
    colors = itertools.cycle(['red', 'blue', 'green', 'orange'])
    markers = itertools.cycle(['o', 's', 'v', '^'])
    
    plt.figure(figsize=(10, 6))
    
    for label, history in histories.items():
        if not history:
            print(f"No data for {label}")
            continue

        steps = sorted(history.keys())
        similarities = [history[step] for step in steps]

        filtered_steps = [step for step in steps if step % interval == 0]
        filtered_similarities = [history[step] for step in filtered_steps]

        color = next(colors)
        marker = next(markers)

        plt.plot(filtered_steps, filtered_similarities,
                 marker=marker, linestyle='-', color=color, label=label)

    plt.xlabel("Step")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity vs. Top Hessian Eigenvector")
    plt.grid(True)
    plt.legend()

    # wandb 上传
    wandb.log({"Cosine similarity (all) vs. Top Hessian Eigenvector": wandb.Image(plt)})

    plt.savefig(save_path)
    print(f"Saved combined plot to {save_path}")
    plt.show()
    
def plot_single_projection_history(history, interval=50, label=None, save_path=None):
    colors = {'batch gradient': 'red', 'full gradient': 'blue', 'gradient difference': 'green'}
    markers = {'batch gradient': 'o', 'full gradient': 's', 'gradient difference': 'v'}
    
    if not history: # 如果没有数据，打印提示信息并返回
        print(f'No data for {label}')
        return
    
    steps = sorted(history.keys())
    projection = [history[step] for step in steps]
    
    filtered_steps = [step for step in steps if step % interval == 0]
    filtered_projection = [history[step] for step in filtered_steps]
    
    # Create figure
    plt.figure(figsize=(10, 6))

    # 绘制投影曲线
    plt.plot(filtered_steps, filtered_projection,
             marker=markers.get(label, 'o'),
             linestyle='-',
             color=colors.get(label, 'black'),
             label=label)

    # 添加标签和标题
    plt.xlabel("Step")
    plt.ylabel("Projection")
    plt.title(f"Projection of {label} onto Hessian's top 10 eigenvectors")
    plt.grid(True)
    plt.legend()
    
    # 上传图像到 wandb
    wandb.log({f"Projection of {label} onto Hessian's top 10 eigenvectors": wandb.Image(plt)})
    
    # 保存图像到指定路径
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # 显示图像
    plt.show()

def plot_projection_history_all(histories, interval=50, save_path=None):
    colors = itertools.cycle(['red', 'blue', 'green', 'orange'])
    markers = itertools.cycle(['o', 's', 'v', '^'])
    
    plt.figure(figsize=(10, 6))
    
    for label, history in histories.items():
        if not history:
            print(f"No data for {label}")
            continue

        steps = sorted(history.keys())
        projection = [history[step] for step in steps]
        
        filtered_steps = [step for step in steps if step % interval == 0]
        filtered_projection = [history[step] for step in filtered_steps]
        
        color = next(colors)
        marker = next(markers)

        plt.plot(filtered_steps, filtered_projection,
                 marker=marker, linestyle='-', color=color, label=label)

    plt.xlabel("Step")
    plt.ylabel("Projection")
    plt.title("Projection onto Hessian's top 10 eigenvectors")
    plt.grid(True)
    plt.legend()
    
    # wandb 上传
    wandb.log({"Projection onto Hessian's top 10 eigenvectors": wandb.Image(plt)})

    plt.savefig(save_path)
    print(f"Saved combined plot to {save_path}")
    plt.show()
    