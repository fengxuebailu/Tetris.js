#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 模型测试脚本
测试不同模型的游戏表现，包括游戏步数、分数和消除行数等指标
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
from copy import deepcopy
from Tetris import shapes, rotate, check, join_matrix, clear_rows

def test_model(model_path, num_games=10, max_steps=200, show_board=False):
    """测试单个模型的性能"""
    try:
        # 尝试使用兼容性加载
        try:
            from model_compatibility import load_old_model
            _, predict_func = load_old_model(model_path)
            # 创建一个类似TetrisAI的接口
            class CompatAI:
                def __init__(self, predict_func):
                    self.predict_func = predict_func
                    
                def predict_move(self, board, piece):
                    from tetris_supervised import TetrisDataCollector
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
            
            ai = CompatAI(predict_func)
            print(f"成功使用兼容模式加载模型: {model_path}")
        except:
            # 如果兼容加载失败，尝试直接加载
            from tetris_supervised import TetrisAI
            ai = TetrisAI(model_path)
            print(f"成功加载AI模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 测试结果统计
    results = {
        "steps": [],      # 每局游戏步数
        "scores": [],     # 每局分数
        "lines": [],      # 每局消除行数
        "game_times": [], # 每局游戏时间
        "terminations": []  # 游戏终止原因
    }
    
    total_start_time = time.time()
    
    for game in range(num_games):
        print(f"\n--- 测试游戏 {game+1}/{num_games} ---")
        board = [[0 for _ in range(10)] for _ in range(20)]
        score = 0
        moves = 0
        lines_cleared = 0
        game_start_time = time.time()
        termination_reason = "达到步数上限"
        
        while moves < max_steps:
            # 随机选择一个方块
            piece = random.choice(shapes)
            move = ai.predict_move(board, piece)
            
            if move is None:
                termination_reason = "AI无法找到有效移动"
                break                # 执行移动
            rotated_piece = deepcopy(piece)
            for _ in range(move['rotation']):
                rotated_piece = rotate(rotated_piece)
            
            # 增强安全检查: 确保移动合法
            # 首先验证初始放置是否有效
            if not check(board, rotated_piece, [move['x'], 0]):
                # 尝试调整x值保持在合法范围内
                adjusted_x = max(-2, min(move['x'], len(board[0]) - len(rotated_piece[0]) + 2))
                if adjusted_x != move['x'] and check(board, rotated_piece, [adjusted_x, 0]):
                    print(f"修正无效的x坐标: {move['x']} → {adjusted_x}")
                    move['x'] = adjusted_x
                else:
                    termination_reason = f"初始位置无效: x={move['x']}, 旋转={move['rotation']}"
                    break
            
            # 计算正确的下落位置y
            move['y'] = 0
            while move['y'] < len(board) and check(board, rotated_piece, [move['x'], move['y']+1]):
                move['y'] += 1
            
            # 最终安全检查
            if not check(board, rotated_piece, [move['x'], move['y']]):
                termination_reason = f"移动无效: x={move['x']}, y={move['y']}, 旋转={move['rotation']}"
                break
                
            join_matrix(board, rotated_piece, [move['x'], move['y']])
            board, cleared = clear_rows(board)
            score += cleared * 100
            lines_cleared += cleared
            moves += 1
            
            # 显示游戏板
            if show_board and moves % 50 == 0:
                print(f"\n游戏板状态 (步数={moves}):")
                for row in board:
                    print(''.join(['□' if cell == 0 else '■' for cell in row]))
            
            # 打印进度
            if moves % 50 == 0:
                print(f"游戏 {game+1}: 已完成 {moves} 步, 当前分数: {score}, 消除行数: {lines_cleared}")
        
        # 记录游戏结果
        game_time = time.time() - game_start_time
        results["steps"].append(moves)
        results["scores"].append(score)
        results["lines"].append(lines_cleared)
        results["game_times"].append(game_time)
        results["terminations"].append(termination_reason)
        
        # 游戏结束统计
        print(f"游戏 {game+1} 结束: 总步数={moves}, 分数={score}, 消除行数={lines_cleared}, 耗时={game_time:.1f}秒")
        print(f"终止原因: {termination_reason}")
    
    # 总测试时间
    total_time = time.time() - total_start_time
    
    # 汇总统计
    summary = {
        "avg_steps": np.mean(results["steps"]),
        "avg_score": np.mean(results["scores"]),
        "avg_lines": np.mean(results["lines"]),
        "avg_time": np.mean(results["game_times"]),
        "max_score": max(results["scores"]),
        "total_time": total_time,
        "termination_counts": {}
    }
    
    # 统计终止原因
    for reason in results["terminations"]:
        if reason in summary["termination_counts"]:
            summary["termination_counts"][reason] += 1
        else:
            summary["termination_counts"][reason] = 1
    
    # 打印汇总结果
    print(f"\n=== 测试结果汇总 ({model_path}) ===")
    print(f"测试局数: {num_games}")
    print(f"平均步数: {summary['avg_steps']:.1f}")
    print(f"平均分数: {summary['avg_score']:.1f}")
    print(f"平均消除行数: {summary['avg_lines']:.1f}")
    print(f"最高分数: {summary['max_score']}")
    print(f"平均每局时间: {summary['avg_time']:.1f}秒")
    print(f"总测试时间: {summary['total_time']:.1f}秒")
    print("终止原因统计:")
    for reason, count in summary["termination_counts"].items():
        print(f"  {reason}: {count}次 ({count/num_games*100:.1f}%)")
    
    return results, summary

def compare_models_performance(model_paths, num_games=5, max_steps=200):
    """比较多个模型的性能"""
    print(f"\n=== 比较模型性能 (每个模型{num_games}局游戏) ===")
    
    results = {}
    summaries = {}
    
    for model_path in model_paths:
        print(f"\n测试模型: {model_path}")
        model_results, model_summary = test_model(model_path, num_games, max_steps)
        results[model_path] = model_results
        summaries[model_path] = model_summary
    
    # 比较结果
    print("\n=== 模型性能比较 ===")
    print(f"{'模型':<20} {'平均步数':<10} {'平均分数':<10} {'平均消除行数':<10}")
    for model_path, summary in summaries.items():
        model_name = model_path.split('/')[-1]
        print(f"{model_name:<20} {summary['avg_steps']:<10.1f} {summary['avg_score']:<10.1f} {summary['avg_lines']:<10.1f}")
    
    # 确定最佳模型
    best_model = max(summaries.items(), key=lambda x: x[1]['avg_score'])
    print(f"\n最佳模型: {best_model[0]}")
    
    # 绘制比较图表
    plot_comparison(results)
    
    return results, summaries

def plot_comparison(results):
    """绘制模型比较图表"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 提取数据
        models = list(results.keys())
        model_names = [m.split('/')[-1] for m in models]
        
        # 平均步数
        avg_steps = [np.mean(results[m]["steps"]) for m in models]
        ax1.bar(model_names, avg_steps)
        ax1.set_title('平均步数')
        ax1.set_ylabel('步数')
        ax1.tick_params(axis='x', rotation=45)
        
        # 平均分数
        avg_scores = [np.mean(results[m]["scores"]) for m in models]
        ax2.bar(model_names, avg_scores)
        ax2.set_title('平均分数')
        ax2.set_ylabel('分数')
        ax2.tick_params(axis='x', rotation=45)
        
        # 平均消除行数
        avg_lines = [np.mean(results[m]["lines"]) for m in models]
        ax3.bar(model_names, avg_lines)
        ax3.set_title('平均消除行数')
        ax3.set_ylabel('行数')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig("model_comparison.png")
        print("比较图表已保存到 model_comparison.png")
        
        # 尝试显示图表
        try:
            plt.show()
        except:
            print("无法显示图表窗口，但图表已保存到文件")
            
    except ImportError:
        print("需要matplotlib库来绘制图表，请运行: pip install matplotlib")
    except Exception as e:
        print(f"绘制图表时出错: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python test_models.py <模型路径1> [模型路径2] ...")
        print("示例: python test_models.py tetris_model.pth tetris_model_epoch_50.pth")
        print("将使用所有现有模型进行测试")
        
        # 使用所有可用的.pth模型文件
        import os
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        
        if not model_files:
            print("错误: 找不到任何模型文件")
            sys.exit(1)
            
        print(f"找到 {len(model_files)} 个模型文件:")
        for i, model_file in enumerate(model_files):
            print(f"{i+1}. {model_file}")
        
        models_to_test = model_files
    else:
        models_to_test = sys.argv[1:]
    
    # 运行比较测试
    print(f"将测试以下模型: {models_to_test}")
    results, summaries = compare_models_performance(models_to_test)
