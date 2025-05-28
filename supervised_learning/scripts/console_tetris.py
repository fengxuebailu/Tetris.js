#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于控制台的俄罗斯方块游戏
"""

import os
import sys
import time
import random
import keyboard
from threading import Thread

# 定义形状
SHAPES = [
    [[1, 1, 1, 1]],           # I
    [[1, 1], [1, 1]],         # O
    [[0, 1, 0], [1, 1, 1]],   # T
    [[1, 0, 0], [1, 1, 1]],   # L
    [[0, 0, 1], [1, 1, 1]],   # J 
    [[1, 1, 0], [0, 1, 1]],   # S
    [[0, 1, 1], [1, 1, 0]],   # Z
]

class ConsoleTetris:
    def __init__(self):
        self.width = 10
        self.height = 20
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.game_over = False
        self.reset()
    
    def reset(self):
        """重置游戏"""
        self.board = [[0] * self.width for _ in range(self.height)]
        self.current_piece = random.choice(SHAPES)
        self.current_x = self.width // 2 - len(self.current_piece[0]) // 2
        self.current_y = 0
        self.score = 0
        self.game_over = False
    
    def draw(self):
        """绘制游戏画面"""
        os.system('cls')  # 清屏
        print(f"得分: {self.score}")
        print("┌" + "─" * (self.width * 2) + "┐")
        
        # 复制游戏板以添加当前方块
        display_board = [row[:] for row in self.board]
        
        # 添加当前方块到显示板
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell and 0 <= self.current_y + y < self.height and 0 <= self.current_x + x < self.width:
                    display_board[self.current_y + y][self.current_x + x] = cell
        
        # 绘制游戏板
        for row in display_board:
            print("│", end="")
            for cell in row:
                print("□ " if cell else "  ", end="")
            print("│")
        
        print("└" + "─" * (self.width * 2) + "┘")
        if self.game_over:
            print("游戏结束!")
    
    def move(self, dx):
        """移动方块"""
        new_x = self.current_x + dx
        if self._is_valid_move(self.current_piece, new_x, self.current_y):
            self.current_x = new_x
            return True
        return False
    
    def rotate(self):
        """旋转方块"""
        rotated = list(zip(*self.current_piece[::-1]))
        if self._is_valid_move(rotated, self.current_x, self.current_y):
            self.current_piece = rotated
            return True
        return False
    
    def _is_valid_move(self, piece, x, y):
        """检查移动是否有效"""
        for i, row in enumerate(piece):
            for j, cell in enumerate(row):
                if cell:
                    if (y + i >= self.height or x + j < 0 or x + j >= self.width or 
                        (y + i >= 0 and self.board[y + i][x + j])):
                        return False
        return True
    
    def _place_piece(self):
        """固定当前方块到游戏板上"""
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell:
                    self.board[self.current_y + i][self.current_x + j] = 1
    
    def update(self):
        """更新游戏状态"""
        if self.game_over:
            return
            
        if self._is_valid_move(self.current_piece, self.current_x, self.current_y + 1):
            self.current_y += 1
        else:
            self._place_piece()
            # 检查和清除完整的行
            lines_cleared = 0
            new_board = []
            for row in self.board:
                if all(row):
                    lines_cleared += 1
                else:
                    new_board.append(row)
            for _ in range(lines_cleared):
                new_board.insert(0, [0] * self.width)
            self.board = new_board
            self.score += lines_cleared * 100
            
            # 生成新方块
            self.current_piece = random.choice(SHAPES)
            self.current_x = self.width // 2 - len(self.current_piece[0]) // 2
            self.current_y = 0
            
            if not self._is_valid_move(self.current_piece, self.current_x, self.current_y):
                self.game_over = True

def main():
    print("=== 俄罗斯方块控制台版 ===")
    print("\n控制说明:")
    print("← → : 左右移动")
    print("↑    : 旋转")
    print("↓    : 加速下落")
    print("Q    : 退出游戏")
    print("\n按任意键开始游戏...")
    input()
    
    game = ConsoleTetris()
    last_update = time.time()
    update_delay = 0.5  # 游戏更新间隔（秒）
    
    while not game.game_over:
        if keyboard.is_pressed('q'):
            break
        
        if keyboard.is_pressed('left'):
            game.move(-1)
        elif keyboard.is_pressed('right'):
            game.move(1)
        elif keyboard.is_pressed('up'):
            game.rotate()
        
        # 如果按下下箭头，加快下落
        current_delay = 0.1 if keyboard.is_pressed('down') else update_delay
        
        if time.time() - last_update >= current_delay:
            game.update()
            last_update = time.time()
        
        game.draw()
        time.sleep(0.05)  # 防止CPU使用率过高
    
    print("\n游戏结束! 最终得分:", game.score)
    input("按回车键退出...")

if __name__ == "__main__":
    main()
