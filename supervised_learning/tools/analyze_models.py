#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 深度模型分析脚本
用于对不同模型进行详细的性能分析
"""

import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 导入Matplotlib配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 设置Python路径以导入模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(PROJECT_ROOT))

from supervised_learning.core.tetris_supervised_fixed import TetrisAI
from Tetris import shapes, check, join_matrix, clear_rows, rotate
from test_models import test_model, compare_models_performance

def analyze_models(model_paths, num_games=10, max_steps=500, detailed=True):
    """深度分析模型性能"""
    print(f"\n=== 深度模型性能分析 (每个模型 {num_games} 局游戏) ===")
    print(f"测试模型: {', '.join(model_paths)}")
    
    # 确保分析结果目录存在
    if not os.path.exists("model_analysis"):
        os.makedirs("model_analysis")
    
    # 测试所有模型
    results, summaries = compare_models_performance(model_paths, num_games, max_steps)
    
    if detailed:
        # 执行详细分析
        perform_detailed_analysis(results, summaries, model_paths)
    
    return results, summaries

def perform_detailed_analysis(results, summaries, model_paths):
    """执行详细的性能分析"""
    
    # 1. 画出每个模型的得分分布
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plot_boxplots([results[m]["scores"] for m in model_paths], 
                  [os.path.basename(m) for m in model_paths], 
                  "各模型得分分布", "得分")
    
    # 2. 画出游戏步数分布
    plt.subplot(2, 2, 2)
    plot_boxplots([results[m]["steps"] for m in model_paths], 
                  [os.path.basename(m) for m in model_paths], 
                  "各模型游戏步数分布", "步数")
    
    # 3. 画出消除行数分布
    plt.subplot(2, 2, 3)
    plot_boxplots([results[m]["lines"] for m in model_paths], 
                  [os.path.basename(m) for m in model_paths], 
                  "各模型消除行数分布", "行数")
    
    # 4. 画出游戏时间分布
    plt.subplot(2, 2, 4)
    plot_boxplots([results[m]["game_times"] for m in model_paths], 
                  [os.path.basename(m) for m in model_paths], 
                  "各模型游戏时间分布", "时间(秒)")
    
    plt.tight_layout()
    plt.savefig("model_analysis/performance_distributions.png")
    
    # 5. 分析终止原因
    plt.figure(figsize=(12, 6))
    termination_data = {}
    for m in model_paths:
        model_name = os.path.basename(m)
        term_reasons = defaultdict(int)
        for reason in results[m]["terminations"]:
            term_reasons[reason] += 1
        termination_data[model_name] = term_reasons
    
    plot_termination_reasons(termination_data, "游戏终止原因分析")
    plt.savefig("model_analysis/termination_reasons.png")
    
    # 6. 计算并显示综合评分
    calculate_composite_scores(summaries)
    
    # 保存详细的结果数据
    save_detailed_results(results, summaries, "model_analysis/detailed_results.txt")
    
    print(f"\n详细分析结果已保存到 model_analysis 目录")

def plot_boxplots(data_lists, labels, title, ylabel):
    """绘制箱线图"""
    plt.boxplot(data_lists)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)

def plot_termination_reasons(termination_data, title):
    """绘制游戏终止原因条形图"""
    models = list(termination_data.keys())
    all_reasons = set()
    for model_reasons in termination_data.values():
        all_reasons.update(model_reasons.keys())
    
    all_reasons = sorted(list(all_reasons))
    x = np.arange(len(models))
    width = 0.8 / len(all_reasons)
    
    for i, reason in enumerate(all_reasons):
        counts = [termination_data[model].get(reason, 0) for model in models]
        plt.bar(x + i * width, counts, width, label=reason)
    
    plt.xlabel('模型')
    plt.ylabel('次数')
    plt.title(title)
    plt.xticks(x + width * len(all_reasons) / 2, models, rotation=45)
    plt.legend()
    plt.tight_layout()

def calculate_composite_scores(summaries):
    """计算综合评分"""
    print("\n=== 模型综合评分 ===")
    print(f"{'模型':<20} {'综合评分':<10} {'计算方法'}")
    
    for model_path, summary in summaries.items():
        model_name = os.path.basename(model_path)
        
        # 综合评分 = 平均分数*0.4 + 平均步数*0.3 + 平均消除行数*0.3
        composite_score = (
            summary['avg_score'] * 0.4 + 
            summary['avg_steps'] * 0.3 + 
            summary['avg_lines'] * 0.3
        )
        
        print(f"{model_name:<20} {composite_score:<10.1f} {'分数×0.4 + 步数×0.3 + 行数×0.3'}")

def save_detailed_results(results, summaries, filename):
    """保存详细的结果到文本文件"""
    with open(filename, 'w') as f:
        f.write("=== 俄罗斯方块监督学习系统 - 模型性能详细分析 ===\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("== 模型汇总比较 ==\n")
        f.write(f"{'模型':<20} {'平均步数':<10} {'平均分数':<10} {'平均消除行数':<10}\n")
        
        for model_path, summary in summaries.items():
            model_name = os.path.basename(model_path)
            f.write(f"{model_name:<20} {summary['avg_steps']:<10.1f} {summary['avg_score']:<10.1f} {summary['avg_lines']:<10.1f}\n")
        
        # 输出详细数据
        f.write("\n\n== 各模型详细数据 ==\n")
        for model_path, result in results.items():
            model_name = os.path.basename(model_path)
            f.write(f"\n= {model_name} 详细数据 =\n")
            
            f.write("\n游戏步数: ")
            f.write(", ".join([str(x) for x in result["steps"]]))
            
            f.write("\n游戏分数: ")
            f.write(", ".join([str(x) for x in result["scores"]]))
            
            f.write("\n消除行数: ")
            f.write(", ".join([str(x) for x in result["lines"]]))
            
            f.write("\n终止原因: ")
            f.write(", ".join(result["terminations"]))
            
            f.write("\n\n统计数据:\n")
            summary = summaries[model_path]
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    f.write(f"- {key}: {value:.2f}\n")
                elif key == "termination_counts":
                    f.write("- 终止原因统计:\n")
                    for reason, count in value.items():
                        f.write(f"  * {reason}: {count}次\n")

def run_benchmark_test(model_paths, deterministic=True):
    """使用特定场景进行基准测试"""
    from tetris_supervised_fixed import TetrisAI
    from Tetris import shapes, rotate, check, join_matrix
    
    print("\n=== 运行特定场景基准测试 ===")
    
    # 创建特定测试场景
    test_board = [[0 for _ in range(10)] for _ in range(20)]
    
    # 添加一些已有方块，创建特定情况
    for j in range(10):
        test_board[19][j] = 1  # 底行填满
    
    for j in range(8):
        test_board[18][j] = 1  # 倒数第二行留两个洞
        
    for i in range(16, 18):
        test_board[i][8] = 1   # 右侧有一个高点
    
    # 测试每个模型
    results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n测试模型: {model_name}")
        
        try:
            ai = TetrisAI(model_path)
            model_result = {"moves": []}
            
            # 对每种方块都进行测试
            for piece_idx, piece in enumerate(shapes):
                board_copy = [row[:] for row in test_board]
                
                # 预测移动
                move = ai.predict_move(board_copy, piece)
                if move:
                    model_result["moves"].append({
                        "piece": piece_idx,
                        "x": move["x"],
                        "rotation": move["rotation"]
                    })
                    print(f"方块 {piece_idx}: x={move['x']}, 旋转={move['rotation']}")
                else:
                    print(f"方块 {piece_idx}: 无法找到有效移动")
            
            results[model_name] = model_result
            
        except Exception as e:
            print(f"测试失败: {e}")
    
    # 比较不同模型的决策
    print("\n=== 基准测试决策比较 ===")
    if len(results) > 1:
        models = list(results.keys())
        for piece_idx in range(len(shapes)):
            print(f"\n方块 {piece_idx} 的决策比较:")
            for model_name in models:
                move = next((m for m in results[model_name]["moves"] if m["piece"] == piece_idx), None)
                if move:
                    print(f"- {model_name}: x={move['x']}, 旋转={move['rotation']}")
                else:
                    print(f"- {model_name}: 无决策")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_models.py <模型路径1> [模型路径2] ...")
        print("示例: python analyze_models.py tetris_model.pth tetris_model_new_full.pth")
        
        # 使用所有可用的.pth模型文件
        import os
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        
        if not model_files:
            print("错误: 找不到任何模型文件")
            sys.exit(1)
            
        print(f"找到 {len(model_files)} 个模型文件:")
        for i, model_file in enumerate(model_files):
            print(f"{i+1}. {model_file}")
        
        print("\n将使用所有模型进行分析...")
        models_to_test = model_files
    else:
        models_to_test = sys.argv[1:]
    
    # 运行全面分析
    print(f"将分析以下模型: {models_to_test}")
    results, summaries = analyze_models(models_to_test, num_games=10)
    
    # 运行基准测试
    benchmark_results = run_benchmark_test(models_to_test)
