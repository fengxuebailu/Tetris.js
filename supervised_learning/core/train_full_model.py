#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 全面模型训练脚本
用于训练更大的新架构模型并保存不同阶段的检查点
"""

import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tetris_supervised_fixed import TetrisNet, TetrisDataset, train_network, save_model

# 导入Matplotlib配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.matplotlibrc import *

def train_full_model(epochs=50, batch_size=64):
    """使用完整数据集训练新架构模型"""
    print("=== 使用完整数据集训练新架构模型 ===")
    
    try:
        # 加载完整训练数据
        print("加载训练数据...")
        game_states, moves = TetrisDataset.load_from_file()
        
        if game_states is None or moves is None or len(game_states) == 0:
            print("错误: 找不到训练数据或数据为空")
            return False
        
        print(f"成功加载训练数据: {len(game_states)} 个样本")
        print(f"数据形状: game_states={game_states.shape}, moves={moves.shape}")
        
        # 创建训练日志目录
        if not os.path.exists("training_logs"):
            os.makedirs("training_logs")
        
        # 指定训练参数
        print(f"\n开始训练，设置参数:")
        print(f"- 轮数: {epochs}")
        print(f"- 批大小: {batch_size}")
        print(f"- 学习率: 0.001 (带调度器)")
        print(f"- 训练集比例: 80%")
        
        # 记录开始时间
        start_time = time.time()
          # 训练网络
        model = train_network(
            game_states, 
            moves, 
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            patience=10  # 10轮没有改进就早停
        )
        
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n训练完成!")
        print(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
          # 保存最终模型
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        final_model_path = os.path.join(models_dir, "tetris_model_new_full.pth")
        save_model(model, final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
        return model
                            
    except Exception as e:
        print(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_training_curves():
    """绘制训练曲线"""
    try:
        # 加载训练历史数据
        import json
        with open("training_logs/training_history.json", "r") as f:
            history = json.load(f)
        
        epochs = range(1, len(history["train_loss"]) + 1)
        
        plt.figure(figsize=(12, 4))
        
        # 训练和验证损失
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], "b-", label="训练损失")
        plt.plot(epochs, history["val_loss"], "r-", label="验证损失")
        plt.title("训练和验证损失")
        plt.xlabel("轮数")
        plt.ylabel("损失")
        plt.legend()
        
        # 学习率变化
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["learning_rates"], "g-")
        plt.title("学习率变化")
        plt.xlabel("轮数")
        plt.ylabel("学习率")
        plt.yscale("log")
        
        plt.tight_layout()
        plt.savefig("training_logs/training_curves.png")
        plt.show()
        
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")

if __name__ == "__main__":
    print("开始全面模型训练流程...")
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 训练模型
    model = train_full_model(epochs=50, batch_size=64)
    
    # 如果成功，绘制训练曲线
    if model:
        plot_training_curves()
        
        # 测试新旧模型
        print("\n测试新旧模型对比:")
        print("运行: python test_models.py tetris_model.pth tetris_model_new_full.pth")
