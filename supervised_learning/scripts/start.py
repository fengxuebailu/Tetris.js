#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 快速启动脚本
提供简单的命令行界面来访问系统的主要功能
"""

import argparse
import os
import sys
import subprocess
from typing import List, Optional

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def print_header():
    """打印程序头部信息"""
    print("=" * 60)
    print("俄罗斯方块监督学习系统")
    print("=" * 60)

def list_available_models() -> List[str]:
    """列出可用的模型文件"""
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    return [f for f in os.listdir(models_dir) if f.endswith('.pth')]

def print_available_models(models: List[str]):
    """打印可用的模型列表"""
    if not models:
        print("没有找到任何模型文件")
        return
    
    print("\n可用的模型:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

def get_model_selection(models: List[str]) -> Optional[str]:
    """让用户选择一个模型"""
    while True:
        try:
            choice = input("\n请选择模型序号 (输入q退出): ")
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print("无效的选择，请重试")
        except ValueError:
            print("请输入有效的数字")

def run_python_script(script_path: str, *args):
    """安全地运行Python脚本"""
    try:
        cmd = [sys.executable, script_path] + list(args)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n脚本执行错误: {e}")
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="俄罗斯方块监督学习系统启动器")
    parser.add_argument('--train', action='store_true', help='训练新模型')
    parser.add_argument('--test', action='store_true', help='测试模型')
    parser.add_argument('--diagnose', action='store_true', help='诊断模型')
    parser.add_argument('--evaluate', action='store_true', help='评估所有模型')
    args = parser.parse_args()

    print_header()

    try:
        if args.train:
            print("\n开始训练流程...")
            run_python_script(
                os.path.join(PROJECT_ROOT, 'core', 'training_pipeline.py'),
                '--collect',
                '--architecture',
                'all'
            )
        
        elif args.test:
            models = list_available_models()
            print_available_models(models)
            if not models:
                return
            
            selected_model = get_model_selection(models)
            if selected_model:
                print(f"\n测试模型: {selected_model}")
                run_python_script(
                    os.path.join(PROJECT_ROOT, 'tools', 'test_models.py'),
                    os.path.join(PROJECT_ROOT, 'models', selected_model)
                )
        
        elif args.diagnose:
            models = list_available_models()
            print_available_models(models)
            if not models:
                return
            
            selected_model = get_model_selection(models)
            if selected_model:
                print(f"\n诊断模型: {selected_model}")
                run_python_script(
                    os.path.join(PROJECT_ROOT, 'tools', 'diagnose_invalid_moves_new.py'),
                    '-m',
                    os.path.join(PROJECT_ROOT, 'models', selected_model)
                )
        
        elif args.evaluate:
            print("\n评估所有模型...")
            run_python_script(
                os.path.join(PROJECT_ROOT, 'tools', 'comprehensive_evaluate.py'),
                '--all'
            )
        
        else:
            print("\n可用的命令:")
            print("1. python start.py --train    # 训练新模型")
            print("2. python start.py --test     # 测试模型")
            print("3. python start.py --diagnose # 诊断模型")
            print("4. python start.py --evaluate # 评估所有模型")
    
    except KeyboardInterrupt:
        print("\n\n操作已取消")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        print("\n完成!")

if __name__ == "__main__":
    main()
