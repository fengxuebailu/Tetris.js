#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pygame 兼容性测试
"""

import os
import sys
import subprocess

def main():
    print("=== Pygame 显示测试 ===")
    print(f"Python 路径: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 设置 Python 环境变量
    env = os.environ.copy()
    env['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    
    # 生成测试代码
    test_code = """
import os
import sys
import pygame

# 初始化 pygame
pygame.init()

# 创建窗口
screen = pygame.display.set_mode((400, 300), pygame.NOFRAME)  # 使用无边框窗口
pygame.display.set_caption('测试窗口')

# 设置颜色
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# 填充背景
screen.fill(WHITE)

# 绘制图形
pygame.draw.rect(screen, RED, (150, 100, 100, 100))

# 更新显示
pygame.display.flip()

# 等待退出
running = True
start_time = pygame.time.get_ticks()
while running and pygame.time.get_ticks() - start_time < 5000:  # 等待5秒
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.time.wait(100)

pygame.quit()
"""
    
    # 不同的显示驱动
    drivers = ['directx', 'windib', 'windows']
    
    for driver in drivers:
        try:
            print(f"\n正在测试 {driver} 驱动...")
            test_env = env.copy()
            test_env['SDL_VIDEODRIVER'] = driver
            
            # 运行测试代码
            process = subprocess.Popen(
                [sys.executable, '-c', test_code],
                env=test_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待进程结束
            stdout, stderr = process.communicate(timeout=10)
            
            if process.returncode == 0:
                print(f"{driver} 驱动测试成功！")
            else:
                print(f"{driver} 驱动测试失败")
                if stdout.strip():
                    print("输出:", stdout.strip())
                if stderr.strip():
                    print("错误:", stderr.strip())
                    
        except subprocess.TimeoutExpired:
            print(f"{driver} 驱动测试超时")
            process.kill()
        except Exception as e:
            print(f"{driver} 驱动测试出错: {e}")

if __name__ == "__main__":
    main()
    input("\n按回车键退出...")
