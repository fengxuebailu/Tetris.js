#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块AI - 训练数据收集脚本
用于收集人类玩家的游戏数据
"""

import os
import sys
import numpy as np
import pygame
import time
from datetime import datetime

# 添加必要的导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Tetris import Tetris

# 定义常量
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_data')
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

class DataCollector:
    def __init__(self):
        self.game = Tetris()
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("俄罗斯方块 - 数据收集模式")
        self.states = []
        self.moves = []
        self.game_count = 0
        
    def collect_data(self, num_games=5):
        """收集指定数量游戏的数据"""
        print("\n=== 俄罗斯方块训练数据收集 ===")
        print("操作说明:")
        print("← →: 左右移动")
        print("↑   : 旋转")
        print("↓   : 快速下落")
        print("ESC : 结束当前游戏")
        print(f"目标：收集{num_games}局游戏数据\n")
        
        self.game_count = 0
        while self.game_count < num_games:
            self._play_one_game()
            
        pygame.quit()
        return np.array(self.states), np.array(self.moves)
    
    def _play_one_game(self):
        """进行一局游戏并收集数据"""
        self.game_count += 1
        self.game.reset()
        print(f"\n开始第 {self.game_count} 局游戏...")
        
        clock = pygame.time.Clock()
        game_over = False
        
        while not game_over:
            # 获取当前状态
            current_state = self.game.get_current_state()
            
            # 处理输入
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                if event.type == pygame.KEYDOWN:
                    move_made = False
                    
                    if event.key == pygame.K_LEFT:
                        if self.game.move(-1):
                            self.states.append(current_state)
                            self.moves.append([1, 0, 0])  # 左移
                            move_made = True
                    elif event.key == pygame.K_RIGHT:
                        if self.game.move(1):
                            self.states.append(current_state)
                            self.moves.append([0, 1, 0])  # 右移
                            move_made = True
                    elif event.key == pygame.K_UP:
                        if self.game.rotate():
                            self.states.append(current_state)
                            self.moves.append([0, 0, 1])  # 旋转
                            move_made = True
                    elif event.key == pygame.K_DOWN:
                        self.game.drop()
                    elif event.key == pygame.K_ESCAPE:
                        print("游戏已终止")
                        game_over = True
                        break
            
            # 更新游戏状态
            self.game.update()
            game_over = self.game.game_over
            
            # 绘制游戏界面
            self.screen.fill((0, 0, 0))
            self.game.draw(self.screen)
            pygame.display.flip()
            
            # 控制游戏速度
            clock.tick(30)
        
        print(f"第 {self.game_count} 局结束，得分：{self.game.score}")
        print(f"当前已收集 {len(self.states)} 个训练样本")
    
    def save_data(self):
        """保存收集的数据"""
        if not self.states:
            print("没有数据可保存！")
            return
        
        save_path = os.path.join(TRAINING_DATA_DIR, 'tetris_training_data.npz')
        np.savez(save_path,
                 states=np.array(self.states),
                 moves=np.array(self.moves))
        print(f"\n数据已保存到: {save_path}")
        print(f"共收集了 {len(self.states)} 个训练样本")

def main():
    collector = DataCollector()
    try:
        collector.collect_data(num_games=5)
        collector.save_data()
    except KeyboardInterrupt:
        print("\n数据收集被中断")
        if collector.states:
            save = input("是否保存已收集的数据？(y/n): ")
            if save.lower() == 'y':
                collector.save_data()
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
