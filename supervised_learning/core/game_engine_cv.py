#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 OpenCV 的俄罗斯方块游戏引擎
"""

import cv2
import numpy as np
import random
import time

shapes = [
    [[1, 1, 1, 1]],           # I
    [[1, 1], [1, 1]],         # O
    [[0, 1, 0], [1, 1, 1]],   # T
    [[1, 0, 0], [1, 1, 1]],   # L
    [[0, 0, 1], [1, 1, 1]],   # J 
    [[1, 1, 0], [0, 1, 1]],   # S
    [[0, 1, 1], [1, 1, 0]],   # Z
]

class TetrisCV:
    """使用OpenCV实现的俄罗斯方块游戏"""
    def __init__(self):
        """初始化游戏"""
        self.width = 400
        self.height = 600
        self.cell_size = 30
        
        self.board_width = 10
        self.board_height = 20
        self.board_x = (self.width - self.board_width * self.cell_size) // 2
        self.board_y = (self.height - self.board_height * self.cell_size) // 2
        
        # 创建游戏窗口
        cv2.namedWindow('Tetris')
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        self.reset()
        print("游戏引擎初始化完成")
    
    def reset(self):
        """重置游戏状态"""
        self.board = [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.current_piece = random.choice(shapes)
        self.current_x = self.board_width // 2 - len(self.current_piece[0]) // 2
        self.current_y = 0
        self.score = 0
        self.game_over = False
    
    def move(self, dx):
        """水平移动方块"""
        new_x = self.current_x + dx
        if self._is_valid_move(self.current_piece, new_x, self.current_y):
            self.current_x = new_x
            return True
        return False
    
    def rotate(self):
        """旋转方块"""
        rotated = list(zip(*self.current_piece[::-1]))  # 矩阵转置
        if self._is_valid_move(rotated, self.current_x, self.current_y):
            self.current_piece = rotated
            return True
        return False
    
    def drop(self):
        """快速下落"""
        while self._is_valid_move(self.current_piece, self.current_x, self.current_y + 1):
            self.current_y += 1
    
    def update(self):
        """更新游戏状态"""
        if self.game_over:
            return
        
        if self._is_valid_move(self.current_piece, self.current_x, self.current_y + 1):
            self.current_y += 1
        else:
            self._place_piece()
            self._clear_lines()
            self.current_piece = random.choice(shapes)
            self.current_x = self.board_width // 2 - len(self.current_piece[0]) // 2
            self.current_y = 0
            
            if not self._is_valid_move(self.current_piece, self.current_x, self.current_y):
                self.game_over = True
    
    def _is_valid_move(self, piece, x, y):
        """检查移动是否有效"""
        for row in range(len(piece)):
            for col in range(len(piece[row])):
                if piece[row][col]:
                    board_x = x + col
                    board_y = y + row
                    if (board_x < 0 or board_x >= self.board_width or
                        board_y >= self.board_height or
                        (board_y >= 0 and self.board[board_y][board_x])):
                        return False
        return True
    
    def _place_piece(self):
        """将当前方块固定到游戏板上"""
        for row in range(len(self.current_piece)):
            for col in range(len(self.current_piece[row])):
                if self.current_piece[row][col]:
                    self.board[self.current_y + row][self.current_x + col] = 1
    
    def _clear_lines(self):
        """清除完整的行"""
        lines_to_clear = []
        for i in range(self.board_height):
            if all(self.board[i]):
                lines_to_clear.append(i)
        
        for line in lines_to_clear:
            del self.board[line]
            self.board.insert(0, [0 for _ in range(self.board_width)])
            self.score += 100
    
    def draw(self):
        """绘制游戏界面"""
        try:
            # 清空画布
            self.canvas.fill(0)
            
            # 绘制游戏板
            for y in range(self.board_height):
                for x in range(self.board_width):
                    if self.board[y][x]:
                        cv2.rectangle(self.canvas,
                                    (self.board_x + x * self.cell_size,
                                     self.board_y + y * self.cell_size),
                                    (self.board_x + (x + 1) * self.cell_size - 1,
                                     self.board_y + (y + 1) * self.cell_size - 1),
                                    (128, 128, 128), -1)
            
            # 绘制当前方块
            for row in range(len(self.current_piece)):
                for col in range(len(self.current_piece[row])):
                    if self.current_piece[row][col]:
                        cv2.rectangle(self.canvas,
                                    (self.board_x + (self.current_x + col) * self.cell_size,
                                     self.board_y + (self.current_y + row) * self.cell_size),
                                    (self.board_x + (self.current_x + col + 1) * self.cell_size - 1,
                                     self.board_y + (self.current_y + row + 1) * self.cell_size - 1),
                                    (255, 255, 255), -1)
            
            # 绘制游戏边框
            cv2.rectangle(self.canvas,
                         (self.board_x - 1, self.board_y - 1),
                         (self.board_x + self.board_width * self.cell_size + 1,
                          self.board_y + self.board_height * self.cell_size + 1),
                         (128, 128, 128), 1)
            
            # 显示分数
            cv2.putText(self.canvas,
                       f'Score: {self.score}',
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1, (255, 255, 255), 2)
            
            if self.game_over:
                cv2.putText(self.canvas,
                           'GAME OVER',
                           (self.width // 4, self.height // 2),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 2)
            
            # 显示画面
            cv2.imshow('Tetris', self.canvas)
            
        except Exception as e:
            print(f"绘制出错: {e}")
            import traceback
            traceback.print_exc()
