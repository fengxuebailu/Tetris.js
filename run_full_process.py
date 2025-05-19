#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 主流程脚本
执行完整的训练、测试和分析流程
"""

import os
import time
import argparse
import subprocess
import sys

def run_command(command, desc, check=True):
    """运行命令并打印输出"""
    print(f"\n{'-'*50}")
    print(f"执行: {desc}")
    print(f"{'-'*50}")
    
    try:
        result = subprocess.run(command, check=check)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"命令执行出错: {e}")
        return False

def main():
    """执行完整流程"""
    parser = argparse.ArgumentParser(description="俄罗斯方块监督学习系统完整流程")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练步骤")
    parser.add_argument("--skip-test", action="store_true", help="跳过测试步骤")
    parser.add_argument("--skip-analysis", action="store_true", help="跳过分析步骤")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--games", type=int, default=20, help="测试游戏数")
    args = parser.parse_args()
    
    # 确保目录存在
    for directory in ["model_backup", "results", "logs"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 记录开始时间
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 创建日志文件
    log_file = f"logs/full_process_{timestamp}.log"
    
    # 备份当前模型
    if not args.skip_train and os.path.exists("tetris_model.pth"):
        print("备份现有模型...")
        backup_path = f"model_backup/tetris_model_{timestamp}.pth"
        try:
            import shutil
            shutil.copy2("tetris_model.pth", backup_path)
            print(f"模型已备份到: {backup_path}")
        except Exception as e:
            print(f"备份模型时出错: {e}")
    
    # 1. 训练模型
    if not args.skip_train:
        print("\n=== 开始模型训练阶段 ===")
        
        # 使用新创建的全面训练脚本
        command = [sys.executable, "train_full_model.py"]
        success = run_command(command, "训练新模型")
        
        if not success:
            print("训练失败，流程终止")
            return
    else:
        print("\n=== 跳过模型训练阶段 ===")
    
    # 2. 测试模型
    if not args.skip_test:
        print("\n=== 开始模型测试阶段 ===")
        
        # 查找要测试的模型
        models_to_test = []
        if os.path.exists("tetris_model.pth"):
            models_to_test.append("tetris_model.pth")
        
        if os.path.exists("tetris_model_new_full.pth"):
            models_to_test.append("tetris_model_new_full.pth")
        
        # 添加最佳模型
        if os.path.exists("tetris_model_best.pth"):
            models_to_test.append("tetris_model_best.pth")
            
        # 如果有epoch模型，添加最后一个
        epoch_models = [f for f in os.listdir('.') if f.startswith('tetris_model_epoch_') and f.endswith('.pth')]
        if epoch_models:
            epoch_models.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if epoch_models[-1] not in models_to_test:
                models_to_test.append(epoch_models[-1])
        
        if models_to_test:
            command = [sys.executable, "test_models.py"] + models_to_test
            run_command(command, f"测试模型: {', '.join(models_to_test)}")
        else:
            print("找不到要测试的模型！")
    else:
        print("\n=== 跳过模型测试阶段 ===")
    
    # 3. 分析模型
    if not args.skip_analysis:
        print("\n=== 开始模型分析阶段 ===")
        
        # 找到所有可用模型
        all_models = [f for f in os.listdir('.') if f.endswith('.pth')]
        
        if all_models:
            # 使用更详细的分析
            command = [sys.executable, "analyze_models.py"] + all_models
            run_command(command, f"深度分析: {', '.join(all_models)}")
            
            # 可视化分析最新的两个模型
            latest_models = sorted(all_models, key=lambda x: os.path.getmtime(x), reverse=True)[:2]
            if len(latest_models) >= 1:
                command = [sys.executable, "visualize_model.py"] + latest_models
                run_command(command, f"可视化分析模型: {', '.join(latest_models)}")
        else:
            print("找不到要分析的模型！")
    else:
        print("\n=== 跳过模型分析阶段 ===")
    
    # 完成所有步骤
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n============================")
    print("完整流程执行完毕！")
    print(f"总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    print("============================")
    
    print(f"\n结果和日志可在以下目录查看:")
    print("- 模型可视化: model_visualization/")
    print("- 性能分析: model_analysis/")
    print("- 训练曲线: training_logs/")
    print("- 模型备份: model_backup/")
    

if __name__ == "__main__":
    main()
