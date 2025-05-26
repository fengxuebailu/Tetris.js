#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 模型兼容性层
用于加载旧版本模型并与新架构兼容
"""

import torch
import torch.nn as nn
import sys
import os

# 确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 动态导入不同架构的模型
try:
    from tetris_supervised import TetrisNet as StandardTetrisNet
    # 尝试导入增强版模型，如果不存在则忽略
    try:
        from enhanced_training import ImprovedTetrisNet
        has_improved_net = True
    except ImportError:
        has_improved_net = False
except ImportError:
    print("警告: 无法导入标准TetrisNet模型")
    StandardTetrisNet = None
    has_improved_net = False

class OldTetrisNet(nn.Module):
    """原始的TetrisNet结构，用于加载旧模型"""
    def __init__(self):
        super(OldTetrisNet, self).__init__()
        # 原始简单网络结构
        self.network = nn.Sequential(
            nn.Linear(216, 128),  # 输入: 板状态(200) + 方块(16)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出: x位置, 旋转角度
        )

    def forward(self, x):
        return self.network(x)

def load_old_model(model_path):
    """加载旧版模型并转换为新架构"""
    print(f"尝试加载模型: {model_path}")
    
    # 根据文件名判断使用哪个网络架构
    model_architecture = "standard"
    if "_enhanced_" in model_path or "_improved_" in model_path:
        model_architecture = "improved"
    
    # 尝试不同架构加载模型
    success = False
    
    # 1. 尝试使用原始架构加载
    try:
        old_model = OldTetrisNet()
        old_model.load_state_dict(torch.load(model_path))
        print(f"成功使用原始架构加载模型: {model_path}")
        success = True
        model_type = "original"
    except Exception as e:
        print(f"使用原始架构加载失败: {e}")
    
    # 2. 尝试使用标准架构加载
    if not success and StandardTetrisNet:
        try:
            old_model = StandardTetrisNet()
            old_model.load_state_dict(torch.load(model_path))
            print(f"成功使用标准架构加载模型: {model_path}")
            success = True
            model_type = "standard"
        except Exception as e:
            print(f"使用标准架构加载失败: {e}")
    
    # 3. 尝试使用增强架构加载
    if not success and has_improved_net and model_architecture == "improved":
        try:
            old_model = ImprovedTetrisNet()
            old_model.load_state_dict(torch.load(model_path))
            print(f"成功使用增强架构加载模型: {model_path}")
            success = True
            model_type = "improved"
        except Exception as e:
            print(f"使用增强架构加载失败: {e}")
    
    if not success:
        raise ValueError(f"无法加载模型 {model_path}，请检查模型格式或文件路径")
    
    # 使用旧模型进行推理
    def predict_move(board, piece, create_state_vector_fn):
        """使用模型预测最佳移动"""
        state_vector = create_state_vector_fn(board, piece)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        # 使用模型预测
        with torch.no_grad():
            output = old_model(state_tensor)
        
        # 输出为 [x位置, 旋转] 
        x = int(round(output[0][0].item()))
        rotation = int(round(output[0][1].item())) % 4  # 确保在0-3范围内
        
        return x, rotation
    
    return old_model, predict_move, model_type

def compare_models(old_path, new_path=None):
    """比较旧模型和新模型的行为差异"""
    import random
    import numpy as np
    from tetris_supervised import TetrisAI
    from Tetris import shapes
    from copy import deepcopy
    
    # 加载旧模型
    old_model, predict_old, model_type = load_old_model(old_path)
    print(f"成功加载模型 {old_path} (类型: {model_type})")
    
    # 如果提供了新模型路径，则加载新模型
    if new_path:
        try:
            new_ai = TetrisAI(new_path)
            have_new_model = True
        except Exception as e:
            print(f"无法加载新模型 ({new_path}): {e}")
            have_new_model = False
    else:
        have_new_model = False
    
    # 创建一个简单的测试
    print("\n比较模型行为差异:")
    board = [[0 for _ in range(10)] for _ in range(20)]
    
    # 测试10个随机方块
    for i in range(10):
        piece = random.choice(shapes)
        piece_array = np.array(piece)
        h, w = piece_array.shape
        
        # 展示当前方块
        print(f"\n测试方块 {i+1}:")
        for row in piece:
            print(''.join(['□' if cell == 0 else '■' for cell in row]))
        
        # 预测旧模型
        try:
            # 手动创建状态向量
            def create_state_vector(board, piece):
                from tetris_supervised import TetrisDataCollector
                collector = TetrisDataCollector()
                return collector.create_state_vector(board, piece)
            
            x_old, rot_old = predict_old(board, piece, create_state_vector)
            print(f"旧模型预测: x={x_old}, 旋转={rot_old}")
            
            # 如果有新模型，也进行预测
            if have_new_model:
                move_new = new_ai.predict_move(board, piece)
                print(f"新模型预测: x={move_new['x']}, 旋转={move_new['rotation']}")
                
                # 显示差异
                if x_old != move_new['x'] or rot_old != move_new['rotation']:
                    print("⚠️ 两个模型预测不一致")
                else:
                    print("✓ 两个模型预测一致")
        except Exception as e:
            print(f"预测出错: {e}")
            import traceback
            traceback.print_exc()
    
    return old_model

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python model_compatibility.py <旧模型路径> [新模型路径]")
        sys.exit(1)
    
    old_model_path = sys.argv[1]
    new_model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    old_model = compare_models(old_model_path, new_model_path)
    
    # 可以保存转换后的模型
    # torch.save(new_model.state_dict(), "converted_model.pth")
