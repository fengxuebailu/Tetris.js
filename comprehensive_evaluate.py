#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 全面模型评估工具
用于综合评估和比较各种模型的有效性、持久性和性能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from copy import deepcopy
import os
import torch
import sys
import pandas as pd
import argparse
from Tetris import shapes, rotate, check, join_matrix, clear_rows

# 确保可以导入必要的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveEvaluator:
    """全面的模型评估器"""
    
    def __init__(self, output_dir="model_evaluation"):
        """初始化评估器"""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.metrics = {}  # 存储所有指标
    
    def load_model(self, model_path):
        """加载模型，支持多种架构"""
        try:
            # 尝试使用兼容性加载
            from model_compatibility import load_old_model
            old_model, predict_func, model_type = load_old_model(model_path)
            
            # 创建一个通用接口
            class GenericAI:
                def __init__(self, predict_func, model_type):
                    self.predict_func = predict_func
                    self.model_type = model_type
                    
                def predict_move(self, board, piece):
                    from tetris_supervised_fixed import TetrisDataCollector
                    collector = TetrisDataCollector()
                    x, rotation = self.predict_func(board, piece, collector.create_state_vector)
                    
                    # 计算最终位置
                    rotated_piece = deepcopy(piece)
                    for _ in range(rotation):
                        rotated_piece = rotate(rotated_piece)
                        
                    # 确保x在合理范围内
                    x = max(-2, min(x, len(board[0]) - len(rotated_piece[0]) + 2))
                    
                    # 计算y (下落位置)
                    y = 0
                    while y < len(board) and check(board, rotated_piece, [x, y+1]):
                        y += 1
                    
                    return {'x': x, 'y': y, 'rotation': rotation}
            
            ai = GenericAI(predict_func, model_type)
            return ai, model_type
            
        except Exception as e:
            print(f"加载模型出错: {e}")
            return None, None
    
    def find_all_models(self):
        """查找工作目录中的所有模型"""
        return [f for f in os.listdir('.') if f.endswith('.pth')]
    
    def evaluate_model(self, model_path, num_games=10, max_steps=1000):
        """全面评估单个模型"""
        print(f"\n=== 全面评估模型: {model_path} ===")
        
        # 加载模型
        ai, model_type = self.load_model(model_path)
        if ai is None:
            print(f"无法加载模型 {model_path}，跳过评估")
            return None
        
        print(f"成功加载模型 (类型: {model_type})")
        print(f"进行 {num_games} 局游戏评估，每局最多 {max_steps} 步")
        
        # 评估指标
        metrics = {
            'steps': [],          # 每局步数
            'scores': [],         # 每局分数
            'lines_cleared': [],  # 每局消除行数
            'game_times': [],     # 每局游戏时间
            'terminations': [],   # 终止原因
            'invalid_moves': [],  # 每局无效移动次数
            'heights': [],        # 最终高度
            'holes': [],          # 最终孔洞数
            'bumpiness': []       # 最终平整度
        }
        
        # 运行多局游戏
        start_time = time.time()
        for game_idx in range(num_games):
            print(f"\n游戏 {game_idx+1}/{num_games}")
            
            # 初始化游戏
            board = [[0 for _ in range(10)] for _ in range(20)]
            score = 0
            steps = 0
            lines = 0
            invalid_moves_count = 0
            game_start_time = time.time()
            termination = "达到最大步数"
            
            # 游戏循环
            while steps < max_steps:
                # 随机选择一个方块
                piece = random.choice(shapes)
                
                # 预测移动
                try:
                    move = ai.predict_move(board, piece)
                    if move is None:
                        termination = "模型无法预测有效移动"
                        break
                    
                    # 准备旋转后的方块
                    rotated_piece = deepcopy(piece)
                    for _ in range(move['rotation']):
                        rotated_piece = rotate(rotated_piece)
                    
                    # 检查移动是否有效
                    if not check(board, rotated_piece, [move['x'], move['y']]):
                        invalid_moves_count += 1
                        # 尝试修复无效移动
                        fixed = False
                        
                        # 方法1: 尝试调整x
                        for test_x in range(max(0, move['x']-2), min(len(board[0])-len(rotated_piece[0])+1, move['x']+3)):
                            if check(board, rotated_piece, [test_x, move['y']]):
                                move['x'] = test_x
                                fixed = True
                                break
                        
                        # 方法2: 如果调整x无效，尝试不同旋转
                        if not fixed:
                            for rot in range(4):
                                if rot == move['rotation']:
                                    continue
                                test_piece = deepcopy(piece)
                                for _ in range(rot):
                                    test_piece = rotate(test_piece)
                                
                                # 尝试找到有效位置
                                for test_x in range(len(board[0])-len(test_piece[0])+1):
                                    test_y = 0
                                    while test_y < len(board) and check(board, test_piece, [test_x, test_y+1]):
                                        test_y += 1
                                    
                                    if check(board, test_piece, [test_x, test_y]):
                                        move = {'x': test_x, 'y': test_y, 'rotation': rot}
                                        rotated_piece = test_piece
                                        fixed = True
                                        break
                                
                                if fixed:
                                    break
                        
                        # 如果无法修复，终止游戏
                        if not fixed:
                            termination = f"无效移动: x={move['x']}, y={move['y']}, 旋转={move['rotation']}"
                            break
                    
                    # 放置方块
                    join_matrix(board, rotated_piece, [move['x'], move['y']])
                    
                    # 清除完整行
                    old_board = [row[:] for row in board]
                    board, cleared = clear_rows(board)
                    lines += cleared
                    score += cleared * 100  # 简单的分数计算
                    
                    # 增加步数
                    steps += 1
                    
                    # 每隔一定步数显示进度
                    if steps % 100 == 0:
                        print(f"  步数: {steps}, 分数: {score}, 消除行数: {lines}")
                    
                except Exception as e:
                    print(f"游戏过程中出错: {e}")
                    termination = f"运行错误: {str(e)}"
                    break
            
            # 游戏结束，记录指标
            game_time = time.time() - game_start_time
            metrics['steps'].append(steps)
            metrics['scores'].append(score)
            metrics['lines_cleared'].append(lines)
            metrics['game_times'].append(game_time)
            metrics['terminations'].append(termination)
            metrics['invalid_moves'].append(invalid_moves_count)
            
            # 计算最终游戏板指标
            height = self._get_height(board)
            holes = self._get_holes(board)
            bumpiness = self._get_bumpiness(board)
            
            metrics['heights'].append(height)
            metrics['holes'].append(holes)
            metrics['bumpiness'].append(bumpiness)
            
            # 输出游戏结果
            print(f"  游戏 {game_idx+1} 结束:")
            print(f"  步数: {steps}, 分数: {score}, 消除行数: {lines}")
            print(f"  无效移动: {invalid_moves_count}, 终止原因: {termination}")
            print(f"  最终高度: {height}, 孔洞数: {holes}, 平整度: {bumpiness}")
        
        # 计算总评估时间
        total_time = time.time() - start_time
        
        # 计算汇总指标
        summary = {
            'avg_steps': np.mean(metrics['steps']),
            'avg_score': np.mean(metrics['scores']),
            'avg_lines': np.mean(metrics['lines_cleared']),
            'avg_invalid_moves': np.mean(metrics['invalid_moves']),
            'avg_height': np.mean(metrics['heights']),
            'avg_holes': np.mean(metrics['holes']),
            'avg_bumpiness': np.mean(metrics['bumpiness']),
            'max_score': max(metrics['scores']),
            'max_steps': max(metrics['steps']),
            'total_time': total_time,
            'model_type': model_type,
            'invalid_move_rate': sum(1 for t in metrics['terminations'] if "无效移动" in t) / num_games * 100
        }
        
        # 统计终止原因
        termination_counts = {}
        for t in metrics['terminations']:
            termination_counts[t] = termination_counts.get(t, 0) + 1
        summary['terminations'] = termination_counts
        
        # 将结果存储到评估器中
        self.metrics[model_path] = {
            'detailed': metrics,
            'summary': summary
        }
        
        print(f"\n评估完成! 平均步数: {summary['avg_steps']:.1f}, 平均分数: {summary['avg_score']:.1f}")
        print(f"平均消除行数: {summary['avg_lines']:.1f}, 无效移动率: {summary['invalid_move_rate']:.2f}%")
        
        return self.metrics[model_path]
    
    def _get_height(self, board):
        """计算游戏板的高度"""
        for y in range(len(board)):
            row = board[y]
            if 1 in row:
                return len(board) - y
        return 0
    
    def _get_holes(self, board):
        """计算游戏板中的孔洞数量"""
        holes = 0
        for col in range(len(board[0])):
            block_found = False
            for row in range(len(board)):
                if board[row][col] == 1:
                    block_found = True
                elif block_found and board[row][col] == 0:
                    holes += 1
        return holes
    
    def _get_bumpiness(self, board):
        """计算游戏板的平整度"""
        heights = []
        for col in range(len(board[0])):
            height = 0
            for row in range(len(board)):
                if board[row][col] == 1:
                    height = len(board) - row
                    break
            heights.append(height)
        
        bumpiness = 0
        for i in range(len(heights)-1):
            bumpiness += abs(heights[i] - heights[i+1])
        return bumpiness
    
    def compare_models(self, model_paths=None, num_games=10, max_steps=1000):
        """比较多个模型的性能"""
        if model_paths is None:
            model_paths = self.find_all_models()
            if not model_paths:
                print("错误: 找不到任何模型文件")
                return
        
        print(f"\n=== 比较 {len(model_paths)} 个模型 ===")
        for path in model_paths:
            print(f"  - {path}")
        
        # 逐个评估模型
        for model_path in model_paths:
            self.evaluate_model(model_path, num_games, max_steps)
        
        # 生成比较报告和可视化
        self.generate_comparison_report()
        self.generate_comparison_visualizations()
        
        return self.metrics
    
    def generate_comparison_report(self):
        """生成详细的模型比较报告"""
        if not self.metrics:
            print("错误: 没有可用的评估数据")
            return
        
        print("\n生成模型比较报告...")
        report_path = f"{self.output_dir}/model_comparison_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("===== 俄罗斯方块监督学习系统 - 模型综合评估报告 =====\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本性能表格
            f.write("=== 基本性能指标 ===\n")
            header = "模型名称".ljust(30)
            metrics = ["平均步数", "平均分数", "平均消除行数", "无效移动率", "模型类型"]
            widths = [10, 10, 10, 10, 10]
            
            for i, metric in enumerate(metrics):
                header += metric.ljust(widths[i])
            f.write(header + "\n")
            f.write("-" * 80 + "\n")
            
            # 按平均步数排序
            models = sorted(self.metrics.items(), key=lambda x: x[1]['summary']['avg_steps'], reverse=True)
            
            for model_path, data in models:
                model_name = os.path.basename(model_path)
                summary = data['summary']
                
                row = model_name.ljust(30)
                row += f"{summary['avg_steps']:.1f}".ljust(widths[0])
                row += f"{summary['avg_score']:.1f}".ljust(widths[1])
                row += f"{summary['avg_lines']:.1f}".ljust(widths[2])
                row += f"{summary['invalid_move_rate']:.2f}%".ljust(widths[3])
                row += f"{summary['model_type']}".ljust(widths[4])
                
                f.write(row + "\n")
            
            # 详细指标
            f.write("\n\n=== 详细模型指标 ===\n")
            for model_path, data in models:
                model_name = os.path.basename(model_path)
                summary = data['summary']
                
                f.write(f"\n模型: {model_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"模型类型: {summary['model_type']}\n")
                f.write(f"平均步数: {summary['avg_steps']:.1f}\n")
                f.write(f"平均分数: {summary['avg_score']:.1f}\n")
                f.write(f"最高分数: {summary['max_score']}\n")
                f.write(f"平均消除行数: {summary['avg_lines']:.1f}\n")
                f.write(f"无效移动率: {summary['invalid_move_rate']:.2f}%\n")
                f.write(f"平均无效移动数: {summary['avg_invalid_moves']:.1f}\n")
                f.write(f"平均最终高度: {summary['avg_height']:.1f}\n")
                f.write(f"平均孔洞数: {summary['avg_holes']:.1f}\n")
                f.write(f"平均平整度: {summary['avg_bumpiness']:.1f}\n")
                
                f.write("游戏终止原因:\n")
                for reason, count in summary['terminations'].items():
                    f.write(f"  - {reason}: {count}次\n")
            
            # 综合评估结论
            f.write("\n\n=== 综合评估结论 ===\n")
            
            best_steps_model = max(models, key=lambda x: x[1]['summary']['avg_steps'])[0]
            best_score_model = max(models, key=lambda x: x[1]['summary']['avg_score'])[0]
            best_lines_model = max(models, key=lambda x: x[1]['summary']['avg_lines'])[0]
            best_valid_model = min(models, key=lambda x: x[1]['summary']['invalid_move_rate'])[0]
            
            f.write(f"1. 耐久性最佳的模型: {os.path.basename(best_steps_model)}\n")
            f.write(f"   平均步数: {self.metrics[best_steps_model]['summary']['avg_steps']:.1f}\n\n")
            
            f.write(f"2. 得分最高的模型: {os.path.basename(best_score_model)}\n")
            f.write(f"   平均分数: {self.metrics[best_score_model]['summary']['avg_score']:.1f}\n\n")
            
            f.write(f"3. 消除行数最多的模型: {os.path.basename(best_lines_model)}\n")
            f.write(f"   平均消除行数: {self.metrics[best_lines_model]['summary']['avg_lines']:.1f}\n\n")
            
            f.write(f"4. 有效移动率最高的模型: {os.path.basename(best_valid_model)}\n")
            f.write(f"   无效移动率: {self.metrics[best_valid_model]['summary']['invalid_move_rate']:.2f}%\n\n")
        
        print(f"比较报告已生成: {report_path}")
    
    def generate_comparison_visualizations(self):
        """生成可视化比较图表"""
        if not self.metrics:
            print("错误: 没有可用的评估数据")
            return
        
        print("\n生成模型比较可视化...")
        
        try:
            # 设置风格
            sns.set_style("whitegrid")
            
            # 准备数据
            models = list(self.metrics.keys())
            model_names = [os.path.basename(m) for m in models]
            
            # 创建数据框
            data = {
                '模型': model_names,
                '平均步数': [self.metrics[m]['summary']['avg_steps'] for m in models],
                '平均分数': [self.metrics[m]['summary']['avg_score'] for m in models],
                '平均消除行数': [self.metrics[m]['summary']['avg_lines'] for m in models],
                '无效移动率': [self.metrics[m]['summary']['invalid_move_rate'] for m in models],
                '平均孔洞数': [self.metrics[m]['summary']['avg_holes'] for m in models],
                '平均高度': [self.metrics[m]['summary']['avg_height'] for m in models],
            }
            df = pd.DataFrame(data)
            
            # 1. 主要性能指标对比
            plt.figure(figsize=(15, 10))
            
            # 平均步数
            plt.subplot(2, 2, 1)
            sns.barplot(x='模型', y='平均步数', data=df, palette='Blues_d')
            plt.title('平均游戏步数')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 平均分数
            plt.subplot(2, 2, 2)
            sns.barplot(x='模型', y='平均分数', data=df, palette='Greens_d')
            plt.title('平均游戏分数')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 平均消除行数
            plt.subplot(2, 2, 3)
            sns.barplot(x='模型', y='平均消除行数', data=df, palette='Oranges_d')
            plt.title('平均消除行数')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 无效移动率
            plt.subplot(2, 2, 4)
            sns.barplot(x='模型', y='无效移动率', data=df, palette='Reds_d')
            plt.title('无效移动率 (%)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            perf_chart_path = f"{self.output_dir}/model_performance_{self.timestamp}.png"
            plt.savefig(perf_chart_path)
            plt.close()
            
            # 2. 游戏板状态指标对比
            plt.figure(figsize=(15, 5))
            
            # 平均孔洞数
            plt.subplot(1, 3, 1)
            sns.barplot(x='模型', y='平均孔洞数', data=df, palette='Purples_d')
            plt.title('平均孔洞数')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 平均高度
            plt.subplot(1, 3, 2)
            sns.barplot(x='模型', y='平均高度', data=df, palette='Purples_d')
            plt.title('平均游戏板高度')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 无效移动与步数关系
            plt.subplot(1, 3, 3)
            sns.scatterplot(x='无效移动率', y='平均步数', data=df, s=100, hue='模型')
            plt.title('无效移动率与游戏步数关系')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            board_chart_path = f"{self.output_dir}/model_board_metrics_{self.timestamp}.png"
            plt.savefig(board_chart_path)
            plt.close()
            
            print(f"性能对比图表已保存: {perf_chart_path}")
            print(f"游戏板指标图表已保存: {board_chart_path}")
            
        except Exception as e:
            print(f"生成图表时出错: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="俄罗斯方块模型全面评估工具")
    parser.add_argument('--models', nargs='+', help='要评估的模型文件路径')
    parser.add_argument('--games', type=int, default=10, help='每个模型评估的游戏局数')
    parser.add_argument('--steps', type=int, default=1000, help='每局游戏的最大步数')
    parser.add_argument('--all', action='store_true', help='评估所有找到的模型')
    args = parser.parse_args()
    
    try:
        print("=" * 50)
        print("俄罗斯方块监督学习系统 - 全面模型评估工具")
        print("=" * 50)
        
        evaluator = ComprehensiveEvaluator()
        
        if args.all:
            all_models = evaluator.find_all_models()
            if all_models:
                print(f"找到 {len(all_models)} 个模型文件")
                evaluator.compare_models(all_models, args.games, args.steps)
            else:
                print("错误: 没有找到任何模型文件")
        elif args.models:
            evaluator.compare_models(args.models, args.games, args.steps)
        else:
            print("请指定要评估的模型或使用--all参数评估所有模型")
            print("使用 python comprehensive_evaluate.py -h 查看帮助信息")
    
    except Exception as e:
        print(f"评估过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
