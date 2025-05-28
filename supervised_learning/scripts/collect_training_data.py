#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块AI - 训练数据收集脚本
用于收集人类玩家的游戏数据，包括游戏状态和操作决策
"""

import os
import sys
import numpy as np
import cv2
import time
from datetime import datetime

# 添加必要的导入路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.game_engine_cv import TetrisCV, shapes

# 定义常量
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_data')
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

class DataCollector:
    def __init__(self):
        """初始化数据收集器"""
        try:
            print("初始化 Tetris 游戏引擎...")
            self.game = TetrisCV()
            print("游戏引擎初始化成功")
            
            self.game_states = []  # 存储游戏状态
            self.moves = []        # 存储对应的移动决策
        except Exception as e:
            print(f"初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise    def collect_game_data(self, num_games=5):
        """收集指定数量游戏的数据"""
        print(f"\n=== 开始收集训练数据 ===")
        print("控制说明:")
        print("← → : 左右移动")
        print("↑    : 旋转")
        print("↓    : 快速下落")
        print("ESC  : 结束当前游戏")
        print(f"目标: 收集 {num_games} 局游戏数据\n")
        print("等待游戏窗口响应...")
        
        # 创建窗口
        cv2.namedWindow('Tetris', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tetris', 800, 600)
        
        for game_idx in range(num_games):
            print(f"开始第 {game_idx + 1} 局游戏...")
            self.game.reset()
            game_over = False
            
            while not game_over:
                current_state = np.array(self.game.board)
                  self.game.update()
                self.game.draw()
                
                # 处理按键输入
                key = cv2.waitKey(100) & 0xFF  # 等待100ms
                
                if key == ord('q') or key == 27:  # q 或 ESC 键退出
                    print("用户退出游戏")
                    break
                
                move_made = False
                if key == ord('a') or key == 81:  # a 或 左箭头键
                    if self.game.move(-1):
                        self.game_states.append(current_state)
                        self.moves.append([1, 0, 0])  # 左移
                        move_made = True
                        print("左移")
                elif key == ord('d') or key == 83:  # d 或 右箭头键
                    if self.game.move(1):
                        self.game_states.append(current_state)
                        self.moves.append([0, 1, 0])  # 右移
                        move_made = True
                        print("右移")
                elif key == ord('w') or key == 82:  # w 或 上箭头键
                    if self.game.rotate():
                        self.game_states.append(current_state)
                        self.moves.append([0, 0, 1])  # 旋转
                        move_made = True
                        print("旋转")
                elif key == ord('s') or key == 84:  # s 或 下箭头键
                    self.game.drop()
                    print("快速下落")
                  game_over = self.game.game_over
                
                if game_over:
                    print(f"第 {game_idx + 1} 局结束，得分: {self.game.score}")
                    time.sleep(1)
                    cv2.putText(self.game.canvas,
                              'Press any key to continue',
                              (self.game.width // 4, self.game.height // 2 + 40),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (255, 255, 255), 2)
                    cv2.imshow('Tetris', self.game.canvas)
                    cv2.waitKey(0)  # 等待按键继续
            
            print(f"本局采集到 {len(self.moves)} 个训练样本")
        
    def save_data(self):
        """保存收集到的数据"""
        if not self.game_states:
            print("没有数据可保存!")
            return
            
        save_path = os.path.join(TRAINING_DATA_DIR, 'tetris_training_data.npz')
        np.savez(save_path, 
                 states=np.array(self.game_states),
                 moves=np.array(self.moves))
        print(f"\n数据已保存到: {save_path}")
        print(f"共收集了 {len(self.game_states)} 个训练样本")

def main():
    print("=== 俄罗斯方块AI训练数据收集系统 ===")
    
    try:
        print("正在初始化数据收集器...")
        collector = DataCollector()
        print("初始化完成，开始收集数据...")
        collector.collect_game_data()
        collector.save_data()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            print("正在清理窗口...")
            cv2.destroyAllWindows()
        except:
            pass  # 忽略清理时的错误

if __name__ == "__main__":
    main()
