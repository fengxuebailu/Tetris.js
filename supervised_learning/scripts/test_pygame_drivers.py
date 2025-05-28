import os
import sys
import pygame

def test_pygame():
    print("Python版本:", sys.version)
    print("当前工作目录:", os.getcwd())
    print("\n初始化 pygame...")
    
    # 尝试不同的显示驱动
    drivers = ['windib', 'directx']
    
    for driver in drivers:
        print(f"\n尝试使用 {driver} 驱动...")
        os.environ['SDL_VIDEODRIVER'] = driver
        
        # 初始化之前先确保关闭
        if pygame.get_init():
            pygame.quit()
        
        try:
            pygame.init()
            print("pygame 版本:", pygame.version.ver)
            print("SDL 版本:", pygame.get_sdl_version())
            
            print("\n创建显示窗口...")
            screen = pygame.display.set_mode((400, 300))
            pygame.display.set_caption("Pygame 测试")
            
            # 填充颜色
            screen.fill((255, 255, 255))  # 白色背景
            pygame.draw.rect(screen, (255, 0, 0), (100, 100, 200, 100))  # 红色矩形
            pygame.display.flip()
            
            print("\n窗口创建成功！")
            print("如果您能看到一个白色背景中有红色矩形的窗口，说明 pygame 工作正常。")
            print("按空格键或关闭窗口退出...")
            
            # 事件循环
            running = True
            start_time = pygame.time.get_ticks()
            while running:
                current_time = pygame.time.get_ticks()
                if current_time - start_time > 5000:  # 5秒后自动退出
                    break
                    
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            running = False
                
                pygame.time.wait(100)
            
            print(f"\n{driver} 驱动测试成功！")
            return  # 成功就退出
            
        except Exception as e:
            print(f"\n使用 {driver} 驱动失败: {e}")
            continue
            
    print("\n所有驱动都失败了")

if __name__ == "__main__":
    test_pygame()
    pygame.quit()
    print("\n测试结束")
