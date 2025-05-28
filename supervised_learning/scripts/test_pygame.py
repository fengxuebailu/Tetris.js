import pygame
import time

print("Starting pygame test...")
pygame.init()
print("Pygame initialized")

# 创建一个简单的窗口
screen = pygame.display.set_mode((300, 300))
pygame.display.set_caption("Pygame Test")
print("Window created")

# 设置背景颜色为白色
screen.fill((255, 255, 255))
pygame.display.flip()
print("Screen updated")

# 等待3秒
print("Waiting for 3 seconds...")
time.sleep(3)

pygame.quit()
print("Pygame quit")
