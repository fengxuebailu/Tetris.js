#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成俄罗斯方块AI训练用的合成数据
"""

import os
import sys
import numpy as np
import random
from datetime import datetime

# 添加必要的导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Tetris import Tetris, shapes, rotate, check, clear_rows

# 定义保存路径
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_data')
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

def create_random_board(height=20, width=10, density=0.3):
    """创建一个随机的游戏板状态"""
    board = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            if random.random() < density * (i / height):  # 底部更容易有方块
                board[i][j] = 1
    return board

def generate_move_for_state(board, piece, width=10):
    """为给定的状态生成一个合理的移动"""
    piece_width = len(piece[0])
    max_x = width - piece_width
    
    # 随机选择一个x位置和旋转
    x = random.randint(0, max_x)
    rotation = random.randint(0, len(piece) - 1)
    
    # 确保移动是有效的
    rotated_piece = piece[rotation]
    while not check(board, rotated_piece, x, 0):
        x = random.randint(0, max_x)
        rotation = random.randint(0, len(piece) - 1)
        rotated_piece = piece[rotation]
    
    return x, rotation

def generate_synthetic_data(num_samples=1000):
    """生成合成训练数据"""
    print(f"开始生成{num_samples}个合成训练样本...")
    
    game_states = []
    moves = []
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"已生成 {i} 个样本...")
            
        # 创建随机游戏板
        board = create_random_board()
        
        # 随机选择一个方块
        current_piece = random.choice(shapes)
        
        # 生成移动
        x, rotation = generate_move_for_state(board, current_piece)
        
        # 准备游戏状态特征
        board_state = board.flatten()
        piece_state = np.zeros(16)  # 4x4的最大方块大小
        piece_matrix = current_piece[0]  # 使用未旋转的方块
        h, w = len(piece_matrix), len(piece_matrix[0])
        for i in range(h):
            for j in range(w):
                if piece_matrix[i][j]:
                    piece_state[i * 4 + j] = 1
                    
        # 合并特征
        game_state = np.concatenate([board_state, piece_state])
        
        # 将x位置和旋转编码为移动向量
        move = np.zeros(3)  # [左移, 右移, 旋转]
        if x < 5:  # 如果在左半边，倾向于左移
            move[0] = 1
        else:  # 如果在右半边，倾向于右移
            move[1] = 1
        if rotation > 0:  # 如果需要旋转
            move[2] = 1
            
        game_states.append(game_state)
        moves.append(move)
    
    # 转换为numpy数组
    game_states = np.array(game_states)
    moves = np.array(moves)
    
    # 保存数据
    save_path = os.path.join(TRAINING_DATA_DIR, 'tetris_training_data.npz')
    np.savez(save_path,
             states=game_states,
             moves=moves)
    
    print(f"\n数据生成完成!")
    print(f"保存路径: {save_path}")
    print(f"样本数量: {len(game_states)}")
    print(f"状态维度: {game_states.shape}")
    print(f"移动维度: {moves.shape}")
    
    return game_states, moves

if __name__ == "__main__":
    generate_synthetic_data()
