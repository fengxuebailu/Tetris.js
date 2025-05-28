import os
import sys
import pygame
import platform

def test_display():
    print("系统信息:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    
    print("\nPygame初始化前:")
    print(f"DISPLAY环境变量: {os.environ.get('DISPLAY')}")
    print(f"SDL_VIDEODRIVER: {os.environ.get('SDL_VIDEODRIVER')}")
    
    # 测试不同的视频驱动
    drivers = ['windib', 'directx', 'windows', None]
    
    for driver in drivers:
        try:
            print(f"\n尝试使用驱动: {driver}")
            if driver:
                os.environ['SDL_VIDEODRIVER'] = driver
            
            pygame.init()
            print("Pygame初始化成功")
            print(f"当前显示驱动: {pygame.display.get_driver()}")
            
            print("尝试创建窗口...")
            screen = pygame.display.set_mode((300, 200))
            pygame.display.set_caption(f"测试 - {driver}")
            
            # 绘制一些内容
            screen.fill((255, 255, 255))
            pygame.draw.rect(screen, (255, 0, 0), (100, 50, 100, 100))
            pygame.display.flip()
            
            print("窗口创建成功！按回车键继续...")
            input()
            
        except Exception as e:
            print(f"使用 {driver} 失败: {e}")
            
        finally:
            pygame.quit()
            
if __name__ == "__main__":
    test_display()
