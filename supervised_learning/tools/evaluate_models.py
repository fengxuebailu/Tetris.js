#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 改进的模型评估脚本
用于全面评估模型性能并对比新旧模型
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy
import traceback

# 导入Matplotlib配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 设置Python路径以导入模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(PROJECT_ROOT))

from supervised_learning.core.tetris_supervised_fixed import TetrisAI
from Tetris import shapes, check, join_matrix, clear_rows, rotate
from test_models import test_model

class EnhancedModelEvaluator:
    def __init__(self, output_dir="model_evaluation"):
        """初始化评估器"""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.model_results = {}
    
    def find_all_models(self):
        """查找工作目录中的所有模型"""
        return [f for f in os.listdir('.') if f.endswith('.pth')]
    
    def evaluate_model(self, model_path, num_games=10, max_steps=250):
        """评估单个模型"""
        print(f"\n=== 评估模型: {model_path} ===")
        print(f"使用 {num_games} 局游戏，每局最多 {max_steps} 步")
        
        # 使用test_model函数进行测试并收集结果
        results, summary = test_model(model_path, num_games, max_steps)
        
        # 保存结果
        self.model_results[model_path] = {
            'results': results,
            'summary': summary
        }
        
        return results, summary
    
    def run_model_comparison(self, model_paths=None, num_games=10, max_steps=250):
        """比较多个模型的性能"""
        if not model_paths:
            model_paths = self.find_all_models()
            if not model_paths:
                print("错误: 找不到任何模型文件")
                return
        
        print(f"\n=== 开始模型对比分析 ===")
        print(f"将评估以下模型: {', '.join(model_paths)}")
        
        # 评估所有模型
        for model_path in model_paths:
            self.evaluate_model(model_path, num_games, max_steps)
        
        # 生成对比报告和可视化
        self.generate_comparison_report()
        self.generate_comparison_visualizations()
    
    def generate_comparison_report(self):
        """生成详细的性能对比报告"""
        if not self.model_results:
            print("错误: 没有可用的评估结果")
            return
        
        report_path = f"{self.output_dir}/{self.timestamp}_comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write("===== 俄罗斯方块监督学习系统 - 模型性能分析 =====\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本性能指标
            f.write("=== 基本性能指标 ===\n")
            f.write(f"{'模型':<30} {'平均步数':<10} {'平均分数':<10} {'平均消除行数':<10} {'无效移动率':<12}\n")
            f.write("-" * 80 + "\n")
            
            models = list(self.model_results.keys())
            models.sort(key=lambda m: self.model_results[m]['summary']['avg_steps'], reverse=True)
            
            for model in models:
                summary = self.model_results[model]['summary']
                model_name = os.path.basename(model)
                
                # 计算无效移动率
                terminations = summary.get('termination_counts', {})
                invalid_moves = 0
                for reason, count in terminations.items():
                    if "无效" in reason or "移动无效" in reason:
                        invalid_moves += count
                
                invalid_rate = invalid_moves / len(self.model_results[model]['results']['steps']) * 100
                
                f.write(f"{model_name:<30} {summary['avg_steps']:<10.1f} {summary['avg_score']:<10.1f} "
                       f"{summary['avg_lines']:<10.1f} {invalid_rate:<10.1f}%\n")
            
            f.write("\n\n=== 详细终止原因分析 ===\n")
            for model in models:
                model_name = os.path.basename(model)
                f.write(f"\n模型: {model_name}\n")
                f.write("-" * 40 + "\n")
                
                for reason, count in self.model_results[model]['summary']['termination_counts'].items():
                    f.write(f"  {reason}: {count} 次\n")
            
            f.write("\n\n=== 性能分析结论 ===\n")
            # 找出最佳模型
            best_steps_model = max(models, key=lambda m: self.model_results[m]['summary']['avg_steps'])
            best_score_model = max(models, key=lambda m: self.model_results[m]['summary']['avg_score'])
            best_lines_model = max(models, key=lambda m: self.model_results[m]['summary']['avg_lines'])
            
            f.write(f"平均步数最高的模型: {os.path.basename(best_steps_model)} "
                   f"({self.model_results[best_steps_model]['summary']['avg_steps']:.1f}步)\n")
            
            f.write(f"平均分数最高的模型: {os.path.basename(best_score_model)} "
                   f"({self.model_results[best_score_model]['summary']['avg_score']:.1f}分)\n")
            
            f.write(f"平均消除行数最多的模型: {os.path.basename(best_lines_model)} "
                   f"({self.model_results[best_lines_model]['summary']['avg_lines']:.1f}行)\n")
        
        print(f"比较报告已生成: {report_path}")
    
    def generate_comparison_visualizations(self):
        """生成比较可视化图表"""
        if not self.model_results:
            return
        
        try:
            # 设置风格
            sns.set_style("whitegrid")
            
            # 1. 性能对比图
            plt.figure(figsize=(15, 10))
            
            models = list(self.model_results.keys())
            model_names = [os.path.basename(m) for m in models]
            
            # 平均步数
            plt.subplot(2, 2, 1)
            steps_data = [self.model_results[m]['summary']['avg_steps'] for m in models]
            sns.barplot(x=model_names, y=steps_data)
            plt.title('平均游戏步数')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('步数')
            
            # 平均分数
            plt.subplot(2, 2, 2)
            score_data = [self.model_results[m]['summary']['avg_score'] for m in models]
            sns.barplot(x=model_names, y=score_data)
            plt.title('平均游戏分数')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('分数')
            
            # 平均消除行数
            plt.subplot(2, 2, 3)
            lines_data = [self.model_results[m]['summary']['avg_lines'] for m in models]
            sns.barplot(x=model_names, y=lines_data)
            plt.title('平均消除行数')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('行数')
            
            # 无效移动率
            plt.subplot(2, 2, 4)
            invalid_rates = []
            for m in models:
                terminations = self.model_results[m]['summary'].get('termination_counts', {})
                invalid_moves = sum(count for reason, count in terminations.items() 
                                  if "无效" in reason or "移动无效" in reason)
                total_games = len(self.model_results[m]['results']['steps'])
                invalid_rates.append(invalid_moves / total_games * 100)
                
            sns.barplot(x=model_names, y=invalid_rates)
            plt.title('无效移动率 (%)')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('百分比')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{self.timestamp}_performance_comparison.png")
            
            
            # 2. 游戏步数分布
            plt.figure(figsize=(12, 6))
            data = []
            for m in models:
                model_name = os.path.basename(m)
                steps = self.model_results[m]['results']['steps']
                for step in steps:
                    data.append({'模型': model_name, '步数': step})
            
            if data:
                df = pd.DataFrame(data)
                sns.boxplot(x='模型', y='步数', data=df)
                plt.title('各模型游戏步数分布')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/{self.timestamp}_steps_distribution.png")
            
            # 3. 消除行数分布
            plt.figure(figsize=(12, 6))
            data = []
            for m in models:
                model_name = os.path.basename(m)
                lines = self.model_results[m]['results']['lines']
                for line in lines:
                    data.append({'模型': model_name, '消除行数': line})
            
            if data:
                df = pd.DataFrame(data)
                sns.boxplot(x='模型', y='消除行数', data=df)
                plt.title('各模型消除行数分布')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/{self.timestamp}_lines_distribution.png")
                
            print(f"比较可视化已保存到 {self.output_dir} 目录")
            
        except Exception as e:
            print(f"生成可视化时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def run_head_to_head_comparison(self, model1, model2, num_games=20):
        """使用相同随机种子进行直接对比测试"""
        print(f"\n=== 进行头对头比较: {model1} vs {model2} ===")
        
        # 加载模型
        from tetris_supervised_fixed import TetrisAI
        try:
            ai1 = TetrisAI(model1)
            ai2 = TetrisAI(model2)
        except Exception as e:
            print(f"加载模型失败: {e}")
            return
            
        # 设置随机种子使两个模型面对相同的游戏场景
        results = {
            model1: {'steps': [], 'scores': [], 'lines': [], 'wins': 0},
            model2: {'steps': [], 'scores': [], 'lines': [], 'wins': 0}
        }
        
        # 运行多次测试
        for game in range(num_games):
            print(f"\n游戏 {game+1}/{num_games}:")
            # 固定随机种子
            seed = game + 42
            
            # 为每个模型运行相同的游戏
            for idx, (model, ai) in enumerate([(model1, ai1), (model2, ai2)]):
                random.seed(seed)
                board = [[0 for _ in range(10)] for _ in range(20)]
                score = 0
                moves = 0
                lines_cleared = 0
                
                while moves < 200:  # 限制最大步数
                    piece = random.choice(shapes)
                    move = ai.predict_move(board, piece)
                    
                    if not move:
                        print(f"模型 {idx+1} 无法找到有效移动")
                        break
                        
                    rotated_piece = deepcopy(piece)
                    for _ in range(move['rotation']):
                        rotated_piece = rotate(rotated_piece)
                        
                    if not check(board, rotated_piece, [move['x'], move['y']]):
                        print(f"模型 {idx+1} 生成了无效移动")
                        break
                        
                    join_matrix(board, rotated_piece, [move['x'], move['y']])
                    board, cleared = clear_rows(board)
                    score += cleared * 100
                    lines_cleared += cleared
                    moves += 1
                
                # 记录结果
                print(f"模型 {idx+1} ({os.path.basename(model)}): {moves}步, {score}分, {lines_cleared}行")
                results[model]['steps'].append(moves)
                results[model]['scores'].append(score)
                results[model]['lines'].append(lines_cleared)
            
            # 确定这局游戏的获胜者
            if results[model1]['steps'][-1] > results[model2]['steps'][-1]:
                results[model1]['wins'] += 1
                winner = 1
            elif results[model2]['steps'][-1] > results[model1]['steps'][-1]:
                results[model2]['wins'] += 1
                winner = 2
            else:
                winner = 0  # 平局
                
            print(f"游戏 {game+1} 获胜者: {'平局' if winner == 0 else f'模型 {winner}'}")
        
        # 生成对比报告
        report_path = f"{self.output_dir}/{self.timestamp}_head_to_head_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"===== 俄罗斯方块 AI 头对头比赛 =====\n")
            f.write(f"模型1: {os.path.basename(model1)}\n")
            f.write(f"模型2: {os.path.basename(model2)}\n")
            f.write(f"游戏局数: {num_games}\n\n")
            
            f.write(f"模型1获胜次数: {results[model1]['wins']} ({results[model1]['wins']/num_games*100:.1f}%)\n")
            f.write(f"模型2获胜次数: {results[model2]['wins']} ({results[model2]['wins']/num_games*100:.1f}%)\n")
            f.write(f"平局次数: {num_games - results[model1]['wins'] - results[model2]['wins']}\n\n")
            
            f.write("平均统计:\n")
            f.write(f"模型1平均步数: {np.mean(results[model1]['steps']):.1f}\n")
            f.write(f"模型2平均步数: {np.mean(results[model2]['steps']):.1f}\n")
            f.write(f"模型1平均分数: {np.mean(results[model1]['scores']):.1f}\n")
            f.write(f"模型2平均分数: {np.mean(results[model2]['scores']):.1f}\n")
            f.write(f"模型1平均消除行数: {np.mean(results[model1]['lines']):.1f}\n")
            f.write(f"模型2平均消除行数: {np.mean(results[model2]['lines']):.1f}\n\n")
            
            f.write("详细游戏结果:\n")
            for game in range(num_games):
                f.write(f"游戏 {game+1}:\n")
                f.write(f"  模型1: {results[model1]['steps'][game]}步, {results[model1]['scores'][game]}分, "
                       f"{results[model1]['lines'][game]}行\n")
                f.write(f"  模型2: {results[model2]['steps'][game]}步, {results[model2]['scores'][game]}分, "
                       f"{results[model2]['lines'][game]}行\n")
                if results[model1]['steps'][game] > results[model2]['steps'][game]:
                    f.write("  获胜者: 模型1\n\n")
                elif results[model2]['steps'][game] > results[model1]['steps'][game]:
                    f.write("  获胜者: 模型2\n\n")
                else:
                    f.write("  结果: 平局\n\n")
        
        print(f"头对头比较报告已保存到: {report_path}")
        
        # 生成可视化
        try:
            plt.figure(figsize=(12, 8))
            
            # 步数比较
            plt.subplot(2, 1, 1)
            indices = range(num_games)
            plt.plot(indices, results[model1]['steps'], 'b-', label=os.path.basename(model1))
            plt.plot(indices, results[model2]['steps'], 'r-', label=os.path.basename(model2))
            plt.title('每局游戏步数比较')
            plt.xlabel('游戏编号')
            plt.ylabel('步数')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 胜率饼图
            plt.subplot(2, 2, 3)
            labels = [os.path.basename(model1), os.path.basename(model2), '平局']
            sizes = [
                results[model1]['wins'], 
                results[model2]['wins'], 
                num_games - results[model1]['wins'] - results[model2]['wins']
            ]
            colors = ['#66b3ff', '#ff9999', '#99ff99']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
            plt.axis('equal')
            plt.title('获胜比例')
            
            # 平均统计条形图
            plt.subplot(2, 2, 4)
            metrics = ['步数', '分数/10', '行数']
            model1_data = [
                np.mean(results[model1]['steps']),
                np.mean(results[model1]['scores'])/10,  # 除以10使图表更加平衡
                np.mean(results[model1]['lines'])
            ]
            model2_data = [
                np.mean(results[model2]['steps']),
                np.mean(results[model2]['scores'])/10,
                np.mean(results[model2]['lines'])
            ]
            
            x = range(len(metrics))
            width = 0.35
            plt.bar([i - width/2 for i in x], model1_data, width, label=os.path.basename(model1), color='#66b3ff')
            plt.bar([i + width/2 for i in x], model2_data, width, label=os.path.basename(model2), color='#ff9999')
            plt.title('平均性能对比')
            plt.xticks(x, metrics)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{self.timestamp}_head_to_head_comparison.png")
            print(f"对比可视化已保存到: {self.output_dir}/{self.timestamp}_head_to_head_comparison.png")
        except Exception as e:
            print(f"生成可视化时出错: {e}")

def display_help():
    """显示帮助信息"""
    print("俄罗斯方块监督学习系统 - 增强模型评估工具")
    print("用法: python evaluate_models.py [选项]")
    print("\n选项:")
    print("  -h, --help             显示此帮助信息")
    print("  -a, --all              评估所有可用的模型")
    print("  -m, --model MODEL      指定要评估的模型")
    print("  -c, --compare M1 M2    直接对比两个模型的性能")
    print("  -n, --num-games NUM    指定测试游戏局数 (默认: 10)")

if __name__ == "__main__":
    import sys
    import pandas as pd
    
    evaluator = EnhancedModelEvaluator()
    
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        display_help()
    elif sys.argv[1] in ['-a', '--all']:
        num_games = 10
        if len(sys.argv) > 3 and sys.argv[2] in ['-n', '--num-games'] and sys.argv[3].isdigit():
            num_games = int(sys.argv[3])
        evaluator.run_model_comparison(num_games=num_games)
    elif sys.argv[1] in ['-m', '--model']:
        if len(sys.argv) < 3:
            print("错误: 未指定模型路径")
            display_help()
        else:
            model_path = sys.argv[2]
            num_games = 10
            if len(sys.argv) > 4 and sys.argv[3] in ['-n', '--num-games'] and sys.argv[4].isdigit():
                num_games = int(sys.argv[4])
            evaluator.evaluate_model(model_path, num_games)
    elif sys.argv[1] in ['-c', '--compare']:
        if len(sys.argv) < 4:
            print("错误: 需要指定两个模型进行比较")
            display_help()
        else:
            model1 = sys.argv[2]
            model2 = sys.argv[3]
            num_games = 20
            if len(sys.argv) > 5 and sys.argv[4] in ['-n', '--num-games'] and sys.argv[5].isdigit():
                num_games = int(sys.argv[5])
            evaluator.run_head_to_head_comparison(model1, model2, num_games)
    else:
        print("错误: 无效的选项")
        display_help()
