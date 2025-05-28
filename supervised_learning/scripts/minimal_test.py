import pygame
import sys

# 初始化 Pygame
pygame.init()

# 创建窗口
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Pygame 测试')

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# 主循环
running = True
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # 绘制
    screen.fill(WHITE)  # 填充白色背景
    
    # 在屏幕中央绘制一个红色方块
    rect = pygame.Rect(295, 215, 50, 50)
    pygame.draw.rect(screen, RED, rect)
    
    # 更新显示
    pygame.display.flip()
    
# 退出 Pygame
pygame.quit()
sys.exit()
