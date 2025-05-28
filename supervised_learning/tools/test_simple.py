#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块AI - 最小测试脚本
"""

import os
import sys
import torch
import numpy as np
import random

sys.path.append(os.path.dirname(__file__))
from supervised_learning.core.tetris_supervised_fixed import TetrisNet
from Tetris import shapes, rotate, check, join_matrix, clear_rows

def test_model(model_path, num_moves=100):
    """测试训练好的模型"""
    print(f"加载模型: {model_path}")
    
    # 加载模型
    model = TetrisNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 初始化游戏状态
    board = [[0 for _ in range(10)] for _ in range(20)]
    score = 0
    moves = 0
    lines = 0
    game_over = False
    
    while not game_over and moves < num_moves:
        # 选择新方块
        piece = random.choice(shapes)
        
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
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取模型预测
        with torch.no_grad():
            action = model(state_tensor)[0]
            
        # 解析动作
        x_pos = action[0].item()  # 0-1范围
        rotation = action[1].item()  # 0-1范围
        
        # 执行动作
        # 1. 旋转
        rotated_piece = piece
        rotation_times = int(rotation * 4) % 4
        for _ in range(rotation_times):
            rotated_piece = rotate(rotated_piece)
            
        # 2. 确定x位置
        piece_width = len(rotated_piece[0])
        max_x = 10 - piece_width
        x = int(x_pos * (max_x + 1))
        x = max(0, min(x, max_x))
        
        # 3. 下落到底部
        y = 0
        while y < 20 and check(board, rotated_piece, (x, y+1)):
            y += 1
            
        # 4. 检查和执行移动
        if check(board, rotated_piece, (x, y)):
            join_matrix(board, rotated_piece, (x, y))
            board, cleared = clear_rows(board)
            score += cleared * 100
            lines += cleared
            moves += 1
            
            # 打印当前状态
            if cleared > 0 or moves % 10 == 0:
                print(f"\n步数: {moves}, 得分: {score}, 消除行数: {lines}")
                print("当前游戏板:")
                for row in board:
                    print(''.join(['□' if cell == 0 else '■' for cell in row]))
        else:
            game_over = True
            
    print(f"\n游戏结束!")
    print(f"总步数: {moves}")
    print(f"最终得分: {score}")
    print(f"消除行数: {lines}")
    
    return {'moves': moves, 'score': score, 'lines': lines}

def main():
    print("=== 俄罗斯方块AI简单测试 ===")
    
    model_path = 'tetris_model_best.pth'
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
        
    test_model(model_path, num_moves=100)

if __name__ == '__main__':
    main()
