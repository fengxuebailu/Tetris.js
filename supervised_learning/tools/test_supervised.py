#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块AI - 简化版测试脚本
"""

import os
import sys
import torch
import numpy as np
import random
from datetime import datetime

# 添加必要的导入路径
sys.path.append(os.path.dirname(__file__))
from supervised_learning.core.tetris_supervised_fixed import TetrisNet
from Tetris import shapes, rotate, check, join_matrix, clear_rows

class TetrisGame:
    def __init__(self):
        self.board = [[0 for _ in range(10)] for _ in range(20)]
        self.current_piece = random.choice(shapes)
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
    def get_state_vector(self):
        """获取当前状态向量"""
        # 游戏板状态
        board_vector = np.array(self.board).flatten()
        
        # 当前方块状态
        piece_matrix = np.zeros((4, 4))
        piece = np.array(self.current_piece)
        h, w = piece.shape
        start_h = (4 - h) // 2
        start_w = (4 - w) // 2
        piece_matrix[start_h:start_h+h, start_w:start_w+w] = piece
        piece_vector = piece_matrix.flatten()
        
        return np.concatenate([board_vector, piece_vector])
        
    def make_move(self, x_pos, rotation):
        """执行移动"""
        if self.game_over:
            return False
            
        # 旋转方块
        piece = self.current_piece[:]
        rotation_times = int(rotation * 4) % 4  # 确保旋转次数在0-3之间
        for _ in range(rotation_times):
            piece = rotate(piece)
            
        # 计算有效的x位置范围
        piece_width = len(piece[0])
        max_x = 10 - piece_width  # 考虑方块宽度
        
        # 转换x位置到有效范围
        x = int(x_pos * (max_x + 1))  # 将0-1映射到0-max_x
        x = max(0, min(x, max_x))
        
        # 找到最低的有效位置
        y = 0
        while y < 20 and check(self.board, piece, (x, y+1)):
            y += 1
            
        # 检查移动是否有效
        if not check(self.board, piece, (x, y)):
            self.game_over = True
            return False
            
        # 放置方块
        join_matrix(self.board, piece, (x, y))
        
        # 消除完整的行
        self.board, cleared = clear_rows(self.board)
        self.lines_cleared += cleared
        self.score += cleared * 100
        
        # 生成新方块
        self.current_piece = random.choice(shapes)
        
        # 检查新方块是否可以放置
        x_center = 4
        if not check(self.board, self.current_piece, (x_center, 0)):
            self.game_over = True
            return False
            
        return True
        
    def print_board(self):
        """打印当前游戏板状态"""
        print("\n当前游戏状态:")
        for row in self.board:
            print(''.join(['□' if cell == 0 else '■' for cell in row]))
        print(f"得分: {self.score}, 消除行数: {self.lines_cleared}")

def test_model(model_path, num_games=3, max_moves=200, verbose=True):
    """测试模型性能"""
    print(f"加载模型: {model_path}")
    model = TetrisNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    results = []
    
    for game_idx in range(num_games):
        print(f"\n开始游戏 {game_idx + 1}/{num_games}")
        game = TetrisGame()
        moves = 0
        last_score = 0
        
        while not game.game_over and moves < max_moves:
            # 获取当前状态
            state = game.get_state_vector()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 获取模型预测
            with torch.no_grad():
                action = model(state_tensor)[0]
            
            # 执行移动
            success = game.make_move(action[0].item(), action[1].item())
            
            if success:
                moves += 1
                if game.score > last_score:
                    if verbose:
                        print(f"第 {moves} 步消除了行! 新增得分: {game.score - last_score}")
                        game.print_board()
                    last_score = game.score
                
                # 每20步报告一次进度
                if moves % 20 == 0 and verbose:
                    print(f"已完成 {moves} 步, 当前得分: {game.score}, 消除行数: {game.lines_cleared}")
            else:
                break
        
        results.append({
            'moves': moves,
            'score': game.score,
            'lines': game.lines_cleared
        })
        
        print(f"\n游戏 {game_idx + 1} 结束")
        print(f"总步数: {moves}")
        print(f"最终得分: {game.score}")
        print(f"消除行数: {game.lines_cleared}")
        game.print_board()
    
    # 打印总结
    print("\n=== 测试结果总结 ===")
    avg_moves = sum(r['moves'] for r in results) / len(results)
    avg_score = sum(r['score'] for r in results) / len(results)
    avg_lines = sum(r['lines'] for r in results) / len(results)
    
    print(f"平均步数: {avg_moves:.1f}")
    print(f"平均得分: {avg_score:.1f}")
    print(f"平均消除行数: {avg_lines:.1f}")
    
    return results

def main():
    print("=== 俄罗斯方块AI测试系统 ===")
    
    # 检查模型文件
    model_path = 'tetris_model_best.pth'
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
        
    # 运行测试
    results = test_model(model_path, num_games=3, max_moves=200, verbose=True)

if __name__ == "__main__":
    main()
