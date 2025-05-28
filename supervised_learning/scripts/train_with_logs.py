#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本 - 包含日志记录
"""
import os
import sys
import time
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from supervised_learning.core.tetris_supervised_fixed import TetrisDataCollector, train_network

# 设置日志
def setup_logging():
    # 创建training_logs目录（如果不存在）
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training_logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # 创建日志文件名（使用时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'training_{timestamp}.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def main():
    try:
        print("正在初始化训练环境...")
        # 设置日志
        log_file = setup_logging()
        print(f"日志文件创建在: {log_file}")
        logging.info("=== 开始新的训练会话 ===")
        
        # 创建数据收集器（小规模测试）
        print("创建数据收集器...")
        collector = TetrisDataCollector(num_games=5, max_moves=50, timeout=30)
        print("开始收集训练数据...")
        logging.info("开始收集训练数据...")
        
        # 收集数据
        game_states, moves = collector.collect_data()
        
        if len(game_states) > 0:
            logging.info(f"成功收集了 {len(game_states)} 个训练样本")
            
            # 训练模型（快速训练）
            logging.info("开始训练模型...")
            model = train_network(game_states, moves, num_epochs=20, batch_size=32)
            logging.info("训练完成！")
            
            logging.info(f"训练日志已保存到: {log_file}")
        else:
            logging.error("数据收集失败")
            
    except Exception as e:
        logging.error(f"发生错误: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
