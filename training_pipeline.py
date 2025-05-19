#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 完整训练流水线
集成数据收集、模型训练、评估和持续优化的完整流程
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
import matplotlib.pyplot as plt
import sys
import json
import argparse
from copy import deepcopy

# 确保可以导入所需模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入我们需要的所有组件
try:
    from tetris_supervised_fixed import TetrisDataset, TetrisAI, TetrisNet, save_model
    from enhanced_training import ImprovedTetrisNet, EnhancedDataCollector, EnhancedModelTrainer
    from train_robust_model import RobustModelTrainer
    from diagnose_invalid_moves import diagnose_moves
    from comprehensive_evaluate import ComprehensiveEvaluator
except ImportError as e:
    print(f"导入必要模块出错: {e}")
    print("请确保所有必要的脚本文件在同一目录下")
    sys.exit(1)

class TrainingPipeline:
    """俄罗斯方块AI完整训练流水线"""
    
    def __init__(self, output_dir="training_pipeline"):
        """初始化训练流水线"""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.history = {}  # 存储训练历史
        
    def run_full_pipeline(self, options=None):
        """运行完整的训练流水线"""
        # 默认选项
        default_options = {
            "collect_data": True,       # 是否收集新数据
            "games": 100,               # 收集数据的游戏局数
            "merge_data": True,         # 是否合并已有数据
            "architectures": ["standard", "improved", "robust"],  # 要训练的模型架构
            "epochs": 100,              # 训练轮数
            "batch_size": 64,           # 批次大小
            "evaluate": True,           # 是否评估模型
            "eval_games": 10,           # 评估的游戏局数
            "max_steps": 1000,          # 每局游戏最大步数
            "auto_select": True         # 是否自动选择最佳模型
        }
        
        # 更新选项
        if options:
            default_options.update(options)
        
        opts = default_options
        
        print("=" * 50)
        print("俄罗斯方块监督学习系统 - 完整训练流水线")
        print("=" * 50)
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n=== 配置信息 ===")
        for k, v in opts.items():
            print(f"{k}: {v}")
        
        pipeline_start = time.time()
        trained_models = []
        
        # 第1步: 收集或加载数据
        if opts["collect_data"]:
            print("\n\n" + "=" * 50)
            print("第1步: 收集训练数据")
            print("=" * 50)
            
            # 使用增强型数据收集器
            print(f"使用增强型数据收集器收集 {opts['games']} 局游戏数据")
            collector = EnhancedDataCollector(num_games=opts["games"])
            game_states, moves = collector.collect_enhanced_data()
            
            # 应用数据增强
            collector.apply_data_augmentation()
            game_states, moves = collector.game_states, collector.optimal_moves
            
            # 合并数据(如果需要)
            if opts["merge_data"]:
                try:
                    prev_states, prev_moves = TetrisDataset.load_from_file()
                    if prev_states is not None and len(prev_states) > 0:
                        print(f"合并现有的 {len(prev_states)} 个样本和新收集的 {len(game_states)} 个样本")
                        game_states = np.concatenate([prev_states, game_states])
                        moves = np.concatenate([prev_moves, moves])
                except Exception as e:
                    print(f"合并数据失败: {e}")
            
            # 保存合并后的数据
            TetrisDataset.save_to_file(game_states, moves)
            print(f"共收集和处理了 {len(game_states)} 个训练样本")
        else:
            # 加载已有数据
            print("\n\n" + "=" * 50)
            print("第1步: 加载已有训练数据")
            print("=" * 50)
            
            game_states, moves = TetrisDataset.load_from_file()
            if game_states is None or len(game_states) == 0:
                print("错误: 没有找到可用的训练数据")
                return False
                
            print(f"加载了 {len(game_states)} 个训练样本")
        
        # 第2步: 训练不同架构的模型
        print("\n\n" + "=" * 50)
        print("第2步: 训练多种模型架构")
        print("=" * 50)
        
        for arch in opts["architectures"]:
            print(f"\n--- 训练 {arch} 架构模型 ---")
            
            if arch == "standard":
                # 训练标准模型
                print("训练标准模型架构")
                from train_network import train_network
                
                model = train_network(
                    game_states, moves, 
                    num_epochs=opts["epochs"], 
                    batch_size=opts["batch_size"],
                    model_name="tetris_standard"
                )
                trained_models.append("tetris_standard_best.pth")
                
            elif arch == "improved":
                # 训练增强模型
                print("训练增强模型架构")
                trainer = EnhancedModelTrainer(model_architecture="improved")
                options = {
                    "num_epochs": opts["epochs"],
                    "batch_size": opts["batch_size"],
                    "model_prefix": "tetris_improved"
                }
                model, _ = trainer.train_with_enhanced_options(game_states, moves, options)
                trained_models.append("tetris_improved_best.pth")
                
            elif arch == "robust":
                # 训练健壮模型
                print("训练健壮型模型架构")
                trainer = RobustModelTrainer()
                options = {
                    "num_epochs": opts["epochs"],
                    "batch_size": opts["batch_size"],
                    "model_prefix": "tetris_robust"
                }
                model, _ = trainer.train_robust_model(game_states, moves, options)
                trained_models.append("tetris_robust_best.pth")
                
            else:
                print(f"未知架构: {arch}，跳过")
                
        # 第3步: 诊断无效移动问题
        print("\n\n" + "=" * 50)
        print("第3步: 诊断无效移动问题")
        print("=" * 50)
        
        # 使用诊断工具检查各个模型的无效移动率
        invalid_move_stats = {}
        for model_path in trained_models:
            print(f"\n--- 诊断模型: {model_path} ---")
            stats = diagnose_moves(model_path, 30)  # 使用30个测试用例
            
            if stats:
                valid_rate = stats['valid'] / stats['total'] * 100 if stats['total'] > 0 else 0
                invalid_move_stats[model_path] = {
                    'valid_rate': valid_rate,
                    'valid': stats['valid'],
                    'invalid': stats['invalid'],
                    'total': stats['total']
                }
                print(f"模型 {model_path} 的有效移动率: {valid_rate:.1f}%")
        
        # 第4步: 全面评估模型性能
        if opts["evaluate"]:
            print("\n\n" + "=" * 50)
            print("第4步: 全面评估模型性能")
            print("=" * 50)
            
            evaluator = ComprehensiveEvaluator()
            evaluation_results = evaluator.compare_models(
                trained_models, 
                num_games=opts["eval_games"],
                max_steps=opts["max_steps"]
            )
            
            # 保存评估结果
            self.evaluation_results = evaluation_results
            
            # 第5步: 选择最佳模型
            if opts["auto_select"] and evaluation_results:
                print("\n\n" + "=" * 50)
                print("第5步: 选择最佳模型")
                print("=" * 50)
                
                # 选择标准: 结合耐久性(步数)和有效移动率
                best_model = None
                best_score = -1
                
                for model_path, metrics in evaluation_results.items():
                    avg_steps = metrics['summary']['avg_steps']
                    invalid_rate = metrics['summary']['invalid_move_rate']
                    valid_rate = 100 - invalid_rate
                    
                    # 计算综合得分: 步数 * 有效率
                    # 这会倾向于选择那些既能长时间游戏又很少产生无效移动的模型
                    score = avg_steps * (valid_rate / 100)
                    
                    print(f"模型 {model_path}: 平均步数={avg_steps:.1f}, 有效率={valid_rate:.1f}%, 综合得分={score:.1f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_path
                
                if best_model:
                    # 将最佳模型复制为推荐模型
                    import shutil
                    best_model_copy = "tetris_recommended.pth"
                    shutil.copy2(best_model, best_model_copy)
                    print(f"\n推荐模型: {best_model} (已复制为 {best_model_copy})")
                    print(f"综合得分: {best_score:.1f}")
                else:
                    print("无法确定最佳模型")
        
        # 计算总耗时
        total_time = time.time() - pipeline_start
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n\n" + "=" * 50)
        print("训练流水线完成!")
        print("=" * 50)
        print(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        print(f"训练了 {len(trained_models)} 个模型:")
        for model in trained_models:
            print(f"  - {model}")
            
        # 提示后续步骤
        print("\n要测试训练好的模型，请运行:")
        print("python test_models.py <model_path>")
        print("\n要进行更详细的评估，请运行:")
        print("python comprehensive_evaluate.py --models <model1> <model2> ...")
        
        return True
    
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="俄罗斯方块AI完整训练流水线")
    parser.add_argument('--collect', action='store_true', help='收集新的训练数据')
    parser.add_argument('--no-collect', action='store_true', help='不收集新数据，使用已有数据')
    parser.add_argument('--games', type=int, default=100, help='数据收集的游戏局数')
    parser.add_argument('--no-merge', action='store_true', help='不合并已有数据')
    parser.add_argument('--architecture', choices=['all', 'standard', 'improved', 'robust'], 
                        default='all', help='要训练的模型架构')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=64, help='批次大小')
    parser.add_argument('--no-eval', action='store_true', help='不进行模型评估')
    parser.add_argument('--eval-games', type=int, default=10, help='评估的游戏局数')
    parser.add_argument('--max-steps', type=int, default=1000, help='每局游戏最大步数')
    parser.add_argument('--no-select', action='store_true', help='不自动选择最佳模型')
    
    args = parser.parse_args()
    
    try:
        options = {
            "collect_data": not args.no_collect if args.no_collect else args.collect,
            "games": args.games,
            "merge_data": not args.no_merge,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "evaluate": not args.no_eval,
            "eval_games": args.eval_games,
            "max_steps": args.max_steps,
            "auto_select": not args.no_select
        }
        
        # 设置架构
        if args.architecture == 'all':
            options["architectures"] = ["standard", "improved", "robust"]
        else:
            options["architectures"] = [args.architecture]
        
        # 运行流水线
        pipeline = TrainingPipeline()
        pipeline.run_full_pipeline(options)
        
    except Exception as e:
        print(f"训练流水线出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
