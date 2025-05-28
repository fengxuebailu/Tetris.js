#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块AI - 调试版测试脚本
"""

import os
import sys
import torch
import numpy as np
import random
import traceback

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'supervised_learning'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'supervised_learning/core'))

from tetris_supervised_fixed import TetrisNet
from Tetris import shapes, rotate, check, join_matrix, clear_rows

def print_debug(msg):
    print(f"[DEBUG] {msg}")

def test_model(model_path, num_moves=100):
    """测试训练好的模型"""
    try:
        print(f"加载模型: {model_path}")
        print_debug(f"当前目录: {os.getcwd()}")
        print_debug(f"Python路径: {sys.path}")
        
        # 加载模型
        model = TetrisNet()
        print_debug("创建模型实例成功")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=device)
        print_debug(f"加载权重成功，权重键: {state_dict.keys()}")
        
        model.load_state_dict(state_dict)
        model.to(device)
        print_debug("加载权重到模型成功")
        
        model.eval()
        print_debug("模型设置为评估模式")
        
        # 初始化游戏状态
        board = [[0 for _ in range(10)] for _ in range(20)]
        score = 0
        moves = 0
        lines = 0
        game_over = False
        
        while not game_over and moves < num_moves:
            # 选择新方块
            piece = random.choice(shapes)
            print_debug(f"\n步骤 {moves + 1}: 新方块形状: {len(piece)}x{len(piece[0])}")
            
            # 准备输入状态
            board_state = np.array(board).flatten()
            piece_state = np.zeros(16)  # 4x4的方块状态
            piece_matrix = np.array(piece)
            h, w = piece_matrix.shape
            start_h = (4 - h) // 2
            start_w = (4 - w) // 2
            temp_matrix = np.zeros((4, 4))
            temp_matrix[start_h:start_h+h, start_w:start_w+w] = piece_matrix
            piece_state = temp_matrix.flatten()
            
            # 合并状态
            state = np.concatenate([board_state, piece_state])
            print_debug(f"输入状态形状: {state.shape}")
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            print_debug(f"输入张量形状: {state_tensor.shape}")
            
            # 获取模型预测
            with torch.no_grad():
                action = model(state_tensor)[0]
                print_debug(f"模型输出: {action}")
                
            # 解析动作
            x_pos = action[0].item()  # 0-1范围
            rotation = action[1].item()  # 0-1范围
            print_debug(f"动作: x_pos={x_pos:.3f}, rotation={rotation:.3f}")
            
            # 执行动作
            # 1. 旋转
            rotated_piece = piece
            rotation_times = int(rotation * 4) % 4
            for _ in range(rotation_times):
                rotated_piece = rotate(rotated_piece)
            print_debug(f"旋转后方块形状: {len(rotated_piece)}x{len(rotated_piece[0])}")
                
            # 2. 确定x位置
            piece_width = len(rotated_piece[0])
            max_x = 10 - piece_width
            x = int(x_pos * (max_x + 1))
            x = max(0, min(x, max_x))
            print_debug(f"计算的x位置: {x} (max_x={max_x})")
            
            # 3. 下落到底部
            y = 0
            while y < 20 and check(board, rotated_piece, (x, y+1)):
                y += 1
            print_debug(f"下落位置: y={y}")
                
            # 4. 检查和执行移动
            if check(board, rotated_piece, (x, y)):
                join_matrix(board, rotated_piece, (x, y))
                board, cleared = clear_rows(board)
                score += cleared * 100
                lines += cleared
                moves += 1
                
                print_debug(f"移动有效，消除行数: {cleared}")
                
                # 打印当前状态
                if cleared > 0 or moves % 10 == 0:
                    print(f"\n步数: {moves}, 得分: {score}, 消除行数: {lines}")
                    print("当前游戏板:")
                    for row in board:
                        print(''.join(['□' if cell == 0 else '■' for cell in row]))
            else:
                print_debug("移动无效，游戏结束")
                game_over = True
                
        print(f"\n游戏结束!")
        print(f"总步数: {moves}")
        print(f"最终得分: {score}")
        print(f"消除行数: {lines}")
        
        return {'moves': moves, 'score': score, 'lines': lines}
        
    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()
        return None

def main():
    print("=== 俄罗斯方块AI调试测试 ===")
    
    # 优先使用checkpoints目录下的最佳模型
    model_paths = [
        os.path.join('supervised_learning', 'models', 'checkpoints', 'tetris_model_best.pth'),
        'tetris_model_best.pth'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
            
    if not model_path:
        print("错误：找不到模型文件")
        return
        
    print(f"使用模型: {model_path}")
    test_model(model_path, num_moves=100)

if __name__ == '__main__':
    main()
