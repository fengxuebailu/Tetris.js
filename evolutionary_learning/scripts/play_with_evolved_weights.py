#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入所需模块
from evolutionary_learning.core.Tetris import check, clear_rows
from evolutionary_learning.core.tetris_evolution import TetrisEvolution

def new_board():
    """创建新的游戏板"""
    return [[0 for _ in range(10)] for _ in range(20)]

def new_piece():
    """生成新的方块"""
    from evolutionary_learning.core.Tetris import shapes
    return random.choice(shapes)

def load_weights(weights_file):
    """加载权重文件"""
    try:
        with open(weights_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载权重文件时出错: {e}")
        sys.exit(1)

def play_single_game(weights):
    """使用给定权重玩一局游戏"""
    # 初始化游戏
    board = new_board()
    current_piece = new_piece()
    score = 0
    lines_cleared = 0
    moves = 0
    
    # 创建评估器
    evaluator = TetrisEvolution()
    
    # 开始游戏循环
    while True:
        # 让AI决定下一步移动
        game_over = not evaluator.make_move(board, current_piece, weights)
        
        if game_over:
            break
            
        # 生成新的方块
        current_piece = new_piece()
        
        # 更新计数
        moves += 1
        score = moves * 10  # 简单的分数计算
        
        # 检查并清除完整的行
        board, cleared = clear_rows(board)
        if cleared > 0:
            lines_cleared += cleared
            score += cleared * 100  # 每消除一行加100分
        
        # 打印当前状态
        print(f"\r当前分数: {score} | 已消除行数: {lines_cleared} | 移动次数: {moves}", end='')
    
    # 游戏结束，打印最终结果
    print(f"\n\n游戏结束!")
    print(f"最终分数: {score}")
    print(f"消除行数: {lines_cleared}")
    
    return {
        'score': score,
        'lines_cleared': lines_cleared
    }

def main():
    # 获取权重文件路径
    weights_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_weights_evolved.json')
    
    print("=" * 50)
    print("俄罗斯方块AI演示")
    print("=" * 50)
    
    # 加载权重
    print(f"加载权重文件: {weights_file}")
    weights = load_weights(weights_file)
    
    # 开始游戏
    print("\n开始游戏...")
    stats = play_single_game(weights)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n游戏被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        print("\n演示结束")