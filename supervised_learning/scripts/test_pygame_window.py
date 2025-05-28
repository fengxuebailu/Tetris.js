import pygame
import time

print("正在初始化 pygame...")
pygame.init()

print("驱动信息:")
print(f"当前显示驱动: {pygame.display.get_driver()}")
print("正在创建窗口...")
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("pygame测试")

print("设置颜色...")
screen.fill((255, 255, 255))  # 白色背景
pygame.display.flip()

print("绘制形状...")
pygame.draw.rect(screen, (255, 0, 0), (100, 100, 50, 50))  # 红色方块
pygame.display.flip()

print("等待5秒...")
start_time = time.time()
while time.time() - start_time < 5:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
    pygame.display.flip()
    time.sleep(0.1)

print("清理pygame...")
pygame.quit()
print("测试完成")
