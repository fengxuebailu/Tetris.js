#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 训练启动脚本
"""

import os
import sys
from training_pipeline import TrainingPipeline

def main():
    """主函数"""
    print("=" * 50)
    print("俄罗斯方块监督学习系统")
    print("=" * 50)
    print("正在初始化训练环境...")
    
    # 创建训练管道
    pipeline = TrainingPipeline()
    
    # 运行完整训练流程
    options = {
        "collect_data": True,        # 收集新的训练数据
        "games": 20,                 # 游戏局数
        "merge_data": True,          # 合并已有数据
        "architectures": ["standard"], # 使用标准模型架构
        "epochs": 50,                # 训练轮数
        "batch_size": 64,            # 批次大小
        "evaluate": True,            # 评估模型
        "eval_games": 5,             # 评估游戏局数
        "max_steps": 200            # 每局最大步数
    }
    
    try:
        success = pipeline.run_full_pipeline(options)
        if not success:
            print("训练流程未成功完成")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n训练过程被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
