from Top_k_Dom_search import search_top_k_dominant_space
import torch
import numpy as np
import pyhessian
import swanlab
import time
import logging
import os
from typing import List
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from config import training_config, device
from model import LinearNetwork
from generate_low_rank import generate_low_rank_identity, generate_low_rank_matrix, generative_dataset
from hessian_utils import compute_hessian_eigen, compute_hessian_eigenvalues_pyhessian
from set_logger import setup_colored_logging
from set_seeds import set_seeds
import torch.nn as nn
from train_visualizer import TrainingVisualizer

def train(model, data, label, single_loader, loss_function, optimizer, steps, record_steps, 
          learning_rate, method, device, threshold, seed, visualizer=None):
    """训练函数，增加可视化支持"""
    
    logger = logging.getLogger()
    logger.info(f"🚀 开始训练 - 种子: {seed}, 总步数: {steps}")

    progress_bar = tqdm(range(steps + 1), 
                       desc=f"训练进度 (Seed {seed})", 
                       ncols=120)

    for step in progress_bar:
        output = model(data)
        loss = loss_function(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step in record_steps:
            logger.info(f"📊 步骤 {step}: 开始计算 Hessian 特征值和主导空间")
            
            try:
                # 重新计算损失（为了计算 Hessian）
                model.zero_grad()
                output = model(data)
                loss_for_hessian = loss_function(output, label)
                
                # 计算 Hessian 特征值
                logger.info(f"🔄 步骤 {step}: 开始计算 Hessian 特征值...")
                hessian_start_time = time.time()
                
                eigenvalues, _ = compute_hessian_eigen(
                    loss=loss_for_hessian,
                    params=model.parameters(),
                    top_k=threshold
                )
                """
                eigenvalues = compute_hessian_eigenvalues_pyhessian(
                    model=model,
                    criterion=loss_function,
                    data_loader=single_loader,
                    top_k=threshold,  # 使用配置文件中的 top_k_pca_number
                    device=device
                )
                """
                
                hessian_time = time.time() - hessian_start_time
                
                # 搜索主导空间
                top_k_results, gaps = search_top_k_dominant_space(
                    eigenvalues=eigenvalues.tolist(),
                    method=method
                )
                
                sorted_gaps = sorted(gaps, reverse=True)
                
                # 更新可视化数据
                if visualizer is not None:
                    visualizer.update_data(
                        step=step,
                        eigenvalues=eigenvalues.tolist(),
                        gaps=sorted_gaps,
                        loss=loss.item(),
                        top_k=top_k_results,
                        hessian_time=hessian_time
                    )

                # 创建日志数据
                log_data = {
                    "step": step,  # 明确添加 step
                    "Training_Loss": loss.item(),
                    "Top_k_Dominant_Space": top_k_results,
                    "Hessian_Compute_Time": hessian_time,
                }

                # 记录特征值 - 使用统一前缀 "Eigenvalue/"
                num_eigenvalues_to_plot = min(threshold, len(eigenvalues))  # 最多100个
                for i in range(num_eigenvalues_to_plot):
                    log_data[f"Eigenvalue/λ_{i+1:03d}"] = float(eigenvalues[i])

                # 记录间隙 - 使用统一前缀 "Gap/"
                num_gaps_to_plot = min(threshold, len(sorted_gaps))  # 最多50个间隙
                for i in range(num_gaps_to_plot):
                    log_data[f"Gap/Gap_{i+1:03d}"] = float(sorted_gaps[i])

                # 记录统计指标
                if len(eigenvalues) > 0:
                    log_data.update({
                        "Stats/Max_Eigenvalue": float(eigenvalues[0]),
                        "Stats/Min_Eigenvalue": float(eigenvalues[-1]),
                        "Stats/Eigenvalue_Sum": float(np.sum(eigenvalues[:num_eigenvalues_to_plot])),
                        "Stats/Eigenvalue_Mean": float(np.mean(eigenvalues[:num_eigenvalues_to_plot])),
                        "Stats/Condition_Number": float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] != 0 else float('inf'),
                    })

                if sorted_gaps:
                    log_data.update({
                        "Stats/Max_Gap": float(sorted_gaps[0]),
                        "Stats/Gap_Mean": float(np.mean(sorted_gaps[:10])),
                    })

                # 一次性记录所有数据
                swanlab.log(log_data, step=step)

                logger.info(f"🎯 Top-k 主导空间: {top_k_results}")
                logger.info(f"📈 最大间隙: {sorted_gaps[0] if sorted_gaps else 0:.6f}")
                logger.info(f"📈 前5个间隙: {sorted_gaps[:5] if len(sorted_gaps) >= 5 else sorted_gaps}")

            except Exception as e:
                logger.error(f"❌ 步骤 {step} 计算失败: {str(e)}")
                logger.error(f"🔍 错误类型: {type(e).__name__}")
                # 至少记录损失
                swanlab.log({
                    "step": step,
                    "Training Loss": loss.item(),
                    "Error": str(e)
                })
    
    logger.info("✅ 训练完成!")



def run_single_experiment(seed, num_layer, base_config, use_custom_rank=True):
    """
    运行单个实验，支持自定义rank
    
    Args:
        seed (int): 随机种子
        num_layer (int): 网络层数
        base_config (dict): 基础配置
        use_custom_rank (bool): 是否使用config中的自定义rank
    """
    logger = logging.getLogger()
    
    # 设置随机种子
    set_seeds(seed)
    logger.info(f"🧪 开始实验: 种子={seed}, 层数={num_layer}")
    
    # 创建实验特定的配置
    experiment_config = base_config.copy()
    experiment_config["num_layer"] = num_layer
    experiment_config["torch_seed"] = seed
    experiment_config["np_seed"] = seed

    # 获取rank信息用于命名
    target_rank = experiment_config.get("rank", "default")
    
    # 动态更新 SwanLab 运行名称
    if use_custom_rank:
        experiment_config["swanlab_run_name"] = f"Exp-Seed{seed}-{num_layer}Layer-Rank{target_rank}-Pyhessian"
        plot_dir = f"/home/ouyangzl/BaseLine/Experiment_10_Search_Top_k_Dominant_Space/plots/seed{seed}_layer{num_layer}_rank{target_rank}_Pyhessian"
    else:
        experiment_config["swanlab_run_name"] = f"Exp-Seed{seed}-{num_layer}Layer-RankDefault-Pyhessian"
        plot_dir = f"/home/ouyangzl/BaseLine/Experiment_10_Search_Top_k_Dominant_Space/plots/seed{seed}_layer{num_layer}_rankdefault_Pyhessian"
    
    # 创建可视化器
    os.makedirs(plot_dir, exist_ok=True)
    visualizer = TrainingVisualizer(save_dir=plot_dir)
    
    # 初始化 SwanLab
    swanlab.init(
        project=experiment_config["swanlab_project_name"],
        experiment_name=experiment_config["swanlab_run_name"],
        config=experiment_config,
        reinit=True  # 允许重新初始化
    )
    
    try:
        # 生成数据（带详细信息输出）
        logger.info("🔄 生成训练数据")
        logger.info(f"📊 实验配置: input_dim={experiment_config['input_dim']}, output_dim={experiment_config['output_dim']}")
        if use_custom_rank:
            logger.info(f"🎯 使用自定义rank: {target_rank}")

        data, label = generative_dataset(
            experiment_config["input_dim"], 
            experiment_config["output_dim"],
            use_custom_rank=use_custom_rank
        )

        data = data.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        # 验证生成的数据
        actual_rank = torch.linalg.matrix_rank(label.squeeze(0)).item()
        logger.info(f"✅ 最终标签矩阵rank验证: {actual_rank}")

        # 创建数据加载器
        single_loader = DataLoader(
            TensorDataset(data, label),
            batch_size=1,
            shuffle=False
        )
        
        # 创建模型
        logger.info("🏗️  初始化模型")
        model = LinearNetwork(
            input_dim=experiment_config["input_dim"],
            hidden_dim=experiment_config["hidden_dim"],
            output_dim=experiment_config["output_dim"],
            num_layer=num_layer,
            var=experiment_config["variance"],
            device=device,
        ).to(device)
        
        # 统计参数数量
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🔢 模型总参数数量: {total_params}")
        
        # 创建优化器和损失函数
        loss_function = nn.MSELoss()

        logger.info("Learning Rate: {}".format(experiment_config["learning_rate"]))
        optimizer = torch.optim.SGD(model.parameters(), lr=experiment_config["learning_rate"])
        
        logger.info(f"📊 损失函数: {loss_function.__class__.__name__}")
        logger.info(f"⚙️  优化器: {optimizer.__class__.__name__}, 学习率={experiment_config['learning_rate']}")
        logger.info(f"🔢 Top-k 数量: {experiment_config['top_k_pca_number']}")
        
        # 开始训练
        train(
            model=model,
            data=data,
            label=label,
            single_loader=single_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            steps=experiment_config["steps"],
            record_steps=experiment_config["record_steps"],
            learning_rate=experiment_config["learning_rate"],
            method=experiment_config["method"],
            device=device,
            threshold=experiment_config["top_k_pca_number"],
            seed=seed,
            visualizer=visualizer  # 添加可视化器
        )
        
        # 训练完成后生成图表
        logger.info("📊 生成可视化图表...")
        visualizer.plot_eigenvalues(seed, num_layer)
        visualizer.plot_gaps(seed, num_layer)
        visualizer.plot_summary(seed, num_layer)
        
        # 保存数据到CSV
        logger.info("💾 保存数据到CSV文件...")
        visualizer.save_data_to_csv(seed, num_layer)

        logger.info(f"✅ 实验完成: 种子={seed}, 层数={num_layer}")
        
    except Exception as e:
        logger.error(f"❌ 实验失败: 种子={seed}, 层数={num_layer}, 错误: {str(e)}")
        raise e
    
    finally:
        # 结束当前 SwanLab 运行
        swanlab.finish()

def run_multiple_experiments(use_custom_rank=True):
    """
    运行多个实验的主函数
    """
    logger = logging.getLogger()
    
    # 实验设置
    seeds = [12138]
    num_layers = [2, 3, 4, 5, 6, 7, 8]
    
    total_experiments = len(seeds) * len(num_layers)
    rank_info = training_config.get("rank", "default")

    logger.info(f"🚀 开始批量实验")
    logger.info(f"📊 实验矩阵: {len(seeds)} 个种子 × {len(num_layers)} 个层数 = {total_experiments} 个实验")
    logger.info(f"🎯 使用rank配置: {rank_info if use_custom_rank else 'default(单位矩阵)'}")
    logger.info(f"🌱 种子列表: {seeds}")
    logger.info(f"🏗️  层数列表: {num_layers}")
    
    experiment_count = 0
    successful_experiments = 0
    failed_experiments = 0
    
    # 双重循环进行实验
    for seed in seeds:
        for num_layer in num_layers:
            experiment_count += 1
            
            logger.info("="*80)
            logger.info(f"🧪 实验 {experiment_count}/{total_experiments}: 种子={seed}, 层数={num_layer}, rank={rank_info}")
            logger.info("="*80)
            
            try:
                run_single_experiment(seed, num_layer, training_config, use_custom_rank)
                successful_experiments += 1
                logger.info(f"✅ 实验 {experiment_count} 成功完成")
                
            except Exception as e:
                failed_experiments += 1
                logger.error(f"❌ 实验 {experiment_count} 失败: {str(e)}")
                logger.error(f"🔍 失败的实验: 种子={seed}, 层数={num_layer}, rank={rank_info}")
                continue

    # 实验总结
    logger.info("="*80)
    logger.info("🎯 批量实验完成!")
    logger.info(f"📊 实验统计:")
    logger.info(f"   总实验数: {total_experiments}")
    logger.info(f"   成功: {successful_experiments}")
    logger.info(f"   失败: {failed_experiments}")
    logger.info(f"   成功率: {successful_experiments/total_experiments*100:.1f}%")
    logger.info(f"   使用的rank: {rank_info}")
    logger.info("="*80)

if __name__ == "__main__":
    # 设置日志
    setup_colored_logging()
    logger = logging.getLogger()
    
    logger.info("🎬 实验开始")
    logger.info(f"🖥️  设备: {device}")
    logger.info(f"📋 基础配置: {training_config}")
    
    # 显示实验计划
    record_steps = training_config["record_steps"]
    logger.info(f"📊 Hessian 计算策略: 仅在指定步数计算 ({len(record_steps)} 个检查点)")
    logger.info(f"📝 记录步数范围: {min(record_steps)} -> {max(record_steps)}, 间隔: {training_config['eigenvalue_interval']}")
    
    # 运行所有实验
    run_multiple_experiments(use_custom_rank=True)  # 使用自定义rank
    
    logger.info("🎉 所有实验完成!")