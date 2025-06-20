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
from traditional_data_utils import generate_low_rank_identity, generative_dataset
from hessian_utils import compute_hessian_eigen 
from set_logger import setup_colored_logging
from set_seeds import set_seeds
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
                        top_k=top_k_results
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

if __name__ == "__main__":
    # 设置 SwanLab API Key
    os.environ["SWANLAB_API_KEY"] = training_config["swanlab_api_key"]
    
    # 设置彩色日志
    logger = setup_colored_logging()
    
    logger.info("🎨 初始化彩色日志系统")
    logger.info("🔑 设置 SwanLab API Key")
    logger.info("📈 初始化 SwanLab")

    try:
        # 初始化 swanlab
        swanlab.init(
            project=training_config["swanlab_project_name"],
            workspace="collapsar",
            name=training_config["swanlab_run_name"],
            config={
                "learning_rate": training_config["learning_rate"],
                "input_dim": training_config["input_dim"],
                "hidden_dim": training_config["hidden_dim"],
                "output_dim": training_config["output_dim"],
                "variance": training_config["variance"],
                "top_k_pca_number": training_config["top_k_pca_number"],
                "eigenvalue_interval": training_config["eigenvalue_interval"],
                "method": training_config["method"],
                "num_layer": training_config["num_layer"],
                "device": str(device),
                "epochs": training_config["steps"],
                "record_steps_count": len(training_config["record_steps"])
            }
        )
        logger.info("✅ SwanLab 初始化成功!")
        
    except Exception as e:
        logger.error(f"❌ SwanLab 初始化失败: {str(e)}")
        logger.error("🔧 请检查 API Key 是否正确")
        raise

    # 设置随机种子
    seed = training_config.get("seed", training_config["torch_seed"])
    logger.info(f"🎲 设置随机种子: numpy={training_config['np_seed']}, torch={training_config['torch_seed']}")

    logger.info(f"🖥️  使用设备: {device}")

    # 显示记录策略信息
    record_steps = training_config["record_steps"]
    logger.info(f"📊 Hessian 计算策略: 仅在指定步数计算 ({len(record_steps)} 个检查点)")
    logger.info(f"📝 记录步数范围: {min(record_steps)} -> {max(record_steps)}, 间隔: {training_config['eigenvalue_interval']}")

    logger.info("🔄 生成训练数据")
    data, label = generative_dataset(training_config["input_dim"], training_config["output_dim"])
    data = data.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)
    logger.info(f"📦 数据形状: data={data.shape}, label={label.shape}")

    single_loader = DataLoader(
        TensorDataset(data, label),
        batch_size=1,
        shuffle=False
    )

    seeds = [42, 43, 44, 45, 46]
    num_layers = [2, 3, 4, 5, 6, 7, 8, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    

    logger.info("🏗️  初始化模型")
    model = LinearNetwork(
        input_dim=training_config["input_dim"],
        hidden_dim=trianing_config["hidden_dim"],
        output_dim=training_config["output_dim"],
        num_layer=training_config["num_layer"],  
        var=training_config["variance"],  
        device=device,  
    ).to(device)

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🔢 模型总参数数量: {total_params:,}")

    # 定义损失函数和优化器
    loss_function = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=training_config["learning_rate"])
    
    logger.info("📊 损失函数: MSELoss")
    logger.info(f"⚙️  优化器: SGD, 学习率={training_config['learning_rate']}")

    # 获取配置参数
    top_k = training_config["top_k_pca_number"]
    logger.info(f"🔢 Top-k 数量: {top_k}")

    # 开始训练
    logger.info("=" * 60)
    logger.info("🚀 开始训练!")
    logger.info("=" * 60)
    start_time = time.time()
    
    try:
        train(
            model=model,
            data=data,
            label=label,
            single_loader=single_loader,
            loss_function=loss_function,
            optimizer=optimizer,  
            steps=training_config["steps"],
            record_steps=record_steps,
            learning_rate=training_config["learning_rate"],
            method=training_config["method"],
            device=device,
            threshold=top_k,
            seed=seed,
        )
        
    except KeyboardInterrupt:
        logger.warning("⚠️  训练被用户中断")
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {str(e)}")
        raise
    
    end_time = time.time()
    training_time = end_time - start_time

    logger.info("=" * 60)
    logger.info(f"🎉 训练完成! 总用时: {training_time:.2f} 秒")
    logger.info(f"⚡ 平均每个 Hessian 计算点用时: {training_time/len(record_steps):.2f} 秒")
    logger.info(f"💾 总共记录了 {len(record_steps)} 个检查点")

    # 结束 swanlab 记录
    try:
        swanlab.finish()
        logger.info("📊 SwanLab 记录已保存并上传")
    except Exception as e:
        logger.error(f"❌ SwanLab 记录保存失败: {str(e)}")

    logger.info("✅ 程序执行完成!")