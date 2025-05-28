#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速训练测试脚本
"""
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from supervised_learning.core.tetris_supervised_fixed import TetrisDataCollector, train_network

def main():
    try:
        print("开始数据收集和训练...")
        
        # 创建数据收集器（小规模测试）
        collector = TetrisDataCollector(num_games=5, max_moves=50, timeout=30)
        
        # 收集数据
        game_states, moves = collector.collect_data()
        
        if len(game_states) > 0:
            print(f"收集了 {len(game_states)} 个训练样本")
            
            # 训练模型（快速训练）
            model = train_network(game_states, moves, num_epochs=20, batch_size=32)
            print("训练完成！")
        else:
            print("数据收集失败")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
