#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块游戏核心引擎
"""

import pygame
import random

shapes = [
    [[1, 1, 1, 1]],           # I
    [[1, 1], [1, 1]],         # O
    [[0, 1, 0], [1, 1, 1]],   # T
    [[1, 0, 0], [1, 1, 1]],   # L
    [[0, 0, 1], [1, 1, 1]],   # J 
    [[1, 1, 0], [0, 1, 1]],   # S
    [[0, 1, 1], [1, 1, 0]],   # Z
]

class Tetris:
    """俄罗斯方块游戏"""
    def __init__(self):
        """初始化游戏"""
        print("正在初始化pygame子系统...")
        pygame.display.init()  # 只初始化显示子系统
        pygame.font.init()     # 初始化字体子系统
        
        self.width = 400
        self.height = 600
        
        print("正在创建游戏窗口...")
        # 尝试使用软件渲染
        pygame.display.set_mode((self.width, self.height), pygame.SWSURFACE)
        # 使用HWSURFACE和DOUBLEBUF标志以改善性能
        self.screen = pygame.display.set_mode((self.width, self.height), 
                                           pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("俄罗斯方块")
        print("游戏窗口已创建")
        self.clock = pygame.time.Clock()
        
        # 游戏参数
        self.cell_size = 30
        self.board_width = 10
        self.board_height = 20
        self.board_x = (self.width - self.board_width * self.cell_size) // 2
        self.board_y = (self.height - self.board_height * self.cell_size) // 2
        
        self.reset()
        
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
        
        # 检查是否可以下落
        if self._is_valid_move(self.current_piece, self.current_x, self.current_y + 1):
            self.current_y += 1
        else:
            # 固定方块
            self._place_piece()
            # 检查和清除完整行
            self._clear_lines()
            # 生成新方块
            self.current_piece = random.choice(shapes)
            self.current_x = self.board_width // 2 - len(self.current_piece[0]) // 2
            self.current_y = 0
            
            # 检查游戏结束
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
            self.screen.fill((0, 0, 0))  # 黑色背景
            
            # 绘制游戏板
            for y in range(self.board_height):
                for x in range(self.board_width):
                    if self.board[y][x]:
                        pygame.draw.rect(self.screen, (128, 128, 128),
                                    (self.board_x + x * self.cell_size,
                                     self.board_y + y * self.cell_size,
                                     self.cell_size - 1, self.cell_size - 1))
            
            # 绘制当前方块
            for row in range(len(self.current_piece)):
                for col in range(len(self.current_piece[row])):
                    if self.current_piece[row][col]:
                        pygame.draw.rect(self.screen, (255, 255, 255),
                                    (self.board_x + (self.current_x + col) * self.cell_size,
                                     self.board_y + (self.current_y + row) * self.cell_size,
                                     self.cell_size - 1, self.cell_size - 1))
            
            # 绘制游戏边框
            pygame.draw.rect(self.screen, (128, 128, 128),
                        (self.board_x - 1, self.board_y - 1,
                         self.board_width * self.cell_size + 2,
                         self.board_height * self.cell_size + 2), 1)
            
            # 显示分数
            font = pygame.font.Font(None, 36)
            score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))
            
            if self.game_over:
                game_over_text = font.render('GAME OVER', True, (255, 0, 0))
                text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2))
                self.screen.blit(game_over_text, text_rect)
            
            pygame.display.flip()
            self.clock.tick(30)  # 限制帧率为30fps
            
        except Exception as e:
            print(f"绘制出错: {e}")
            raise
