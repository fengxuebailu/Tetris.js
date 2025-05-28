#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块AI - 简化版监督学习训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from datetime import datetime

# 添加必要的导入路径
sys.path.append(os.path.dirname(__file__))
from supervised_learning.core.tetris_supervised_fixed import TetrisNet, TetrisDataset
from Tetris import shapes, rotate, check, join_matrix, clear_rows

# 定义保存路径
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'supervised_learning', 'models', 'checkpoints')
TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), 'supervised_learning', 'training_data')

# 确保目录存在
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

def train_model(game_states, moves, num_epochs=50, batch_size=64, learning_rate=0.001):    """训练模型的核心函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n开始训练模型...")
    print(f"使用设备: {device}")
    print(f"数据集大小: {len(game_states)} 样本")
    
    # 创建数据集
    dataset = TetrisDataset(game_states, moves)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型和优化器
    model = TetrisNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for states, target_moves in train_loader:
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, target_moves)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, target_moves in val_loader:
                outputs = model(states)
                val_loss += criterion(outputs, target_moves).item()
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'轮数 {epoch+1}/{num_epochs}, '
              f'训练损失: {avg_train_loss:.4f}, '
              f'验证损失: {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss            model_path = os.path.join(MODEL_SAVE_DIR, 'tetris_model_best.pth')
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f"验证损失在{patience}轮内没有改善，停止训练")
            break
              # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, f'tetris_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
    
    print(f"训练完成! 最佳验证损失: {best_val_loss:.4f}")
    return model

def main():
    print("=== 俄罗斯方块AI训练系统 ===")
      # 加载现有数据或使用新数据
    training_data_path = os.path.join(TRAINING_DATA_DIR, 'tetris_training_data.npz')
    if os.path.exists(training_data_path):
        print("发现现有训练数据，正在加载...")
        game_states, moves = TetrisDataset.load_from_file(training_data_path)
        if game_states is None or moves is None:
            print("加载数据失败，请确保数据文件完整")
            return
    else:
        print("未找到训练数据文件")
        return
    
    # 开始训练
    print(f"\n训练数据统计:")
    print(f"样本数量: {len(game_states)}")
    print(f"输入维度: {game_states.shape}")
    print(f"输出维度: {moves.shape}")
    
    model = train_model(game_states, moves)

if __name__ == "__main__":
    main()
