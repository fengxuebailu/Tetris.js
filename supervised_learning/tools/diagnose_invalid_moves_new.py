#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 无效移动诊断工具
用于分析和诊断神经网络预测的无效移动问题
"""

import os
import sys
import time
import random
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# 导入Matplotlib配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 设置Python路径以导入模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(PROJECT_ROOT))

from supervised_learning.core.tetris_supervised_fixed import TetrisAI, TetrisDataCollector
from supervised_learning.core.tetris_supervised import TetrisAI as OldTetrisAI
from Tetris import shapes, rotate, check, join_matrix

# 导入matplotlib全局配置
from tools.matplotlibrc import *

# 诊断输出目录
DIAGNOSTICS_DIR = os.path.join(PROJECT_ROOT, "move_diagnostics")
if not os.path.exists(DIAGNOSTICS_DIR):
    os.makedirs(DIAGNOSTICS_DIR)

def setup_environment():
    """初始化诊断环境"""
    # 创建诊断目录
    if not os.path.exists(DIAGNOSTICS_DIR):
        os.makedirs(DIAGNOSTICS_DIR)
    
    # 时间戳，用于保存诊断文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return timestamp

def load_model(model_path):
    """加载模型，支持多种架构"""
    try:
        # 先尝试增强版AI
        try:
            ai = TetrisAI(model_path)
            print(f"成功加载AI模型 (新架构): {model_path}")
            return ai, "new"
        except Exception as e:
            print(f"使用新架构加载失败: {e}，尝试旧架构...")
            try:
                ai = OldTetrisAI(model_path)
                print(f"成功加载AI模型 (旧架构): {model_path}")
                return ai, "old"
            except Exception as e2:
                print(f"使用旧架构加载也失败: {e2}")
                raise ValueError(f"无法加载模型 {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None

def visualize_move(board, piece, move, idx, timestamp):
    """可视化移动并保存图像"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        
        # 绘制原始游戏板
        ax1.set_title("原始游戏板")
        ax1.imshow(board, cmap='binary', alpha=0.8)
        ax1.grid(True, which='both', color='lightgrey', linestyle='-', alpha=0.5)
        
        # 绘制方块
        piece_array = np.array(piece)
        piece_img = np.zeros((len(board), len(board[0])))
        h, w = piece_array.shape
        offset_y = 1  # 在顶部显示方块
        offset_x = (len(board[0]) - w) // 2  # 在中间显示方块
        for y in range(h):
            for x in range(w):
                if piece_array[y, x] and 0 <= y+offset_y < len(board) and 0 <= x+offset_x < len(board[0]):
                    piece_img[y+offset_y, x+offset_x] = 2
        ax1.imshow(piece_img, cmap='hot', alpha=0.5)
        
        # 绘制预测后的游戏板
        ax2.set_title(f"预测移动: x={move['x']}, y={move['y']}, 旋转={move['rotation']}")
        board_copy = [row[:] for row in board]
        
        # 旋转方块
        rotated_piece = deepcopy(piece)
        for _ in range(move['rotation']):
            rotated_piece = rotate(rotated_piece)
            
        # 检查是否有效
        is_valid = check(board_copy, rotated_piece, [move['x'], move['y']])
        
        # 更新游戏板副本
        if is_valid:
            join_matrix(board_copy, rotated_piece, [move['x'], move['y']])
            ax2.set_title(f"有效移动: x={move['x']}, y={move['y']}, 旋转={move['rotation']}")
        else:
            ax2.set_title(f"无效移动: x={move['x']}, y={move['y']}, 旋转={move['rotation']}")
            # 尝试在游戏板上显示无效方块位置（红色）
            invalid_piece = np.zeros((len(board), len(board[0])))
            for y_rel, row in enumerate(rotated_piece):
                for x_rel, cell in enumerate(row):
                    y_abs = move['y'] + y_rel
                    x_abs = move['x'] + x_rel
                    if cell and 0 <= y_abs < len(board) and 0 <= x_abs < len(board[0]):
                        invalid_piece[y_abs, x_abs] = 3
            ax2.imshow(invalid_piece, cmap='Reds', alpha=0.7)
        
        ax2.imshow(board_copy, cmap='binary', alpha=0.8)
        ax2.grid(True, which='both', color='lightgrey', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        filename = f"{DIAGNOSTICS_DIR}/{timestamp}_move_{idx}.png"
        plt.savefig(filename)
        plt.close()
        return is_valid, filename
    except Exception as e:
        print(f"可视化移动时出错: {e}")
        return None, None

def diagnose_moves(model_path, num_tests=50):
    """诊断指定模型预测的移动"""
    timestamp = setup_environment()
    ai, model_type = load_model(model_path)
    
    if ai is None:
        print("无法加载模型，终止诊断")
        return
    
    print(f"\n=== 开始诊断模型 {model_path} (类型: {model_type}) ===")
    print(f"将进行 {num_tests} 次随机测试")
    
    # 初始化诊断统计
    stats = {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'details': []
    }
    
    # 进行多次随机测试
    for i in range(num_tests):
        print(f"\n测试 {i+1}/{num_tests}:")
        board = [[0 for _ in range(10)] for _ in range(20)]
        
        # 生成一些随机方块，使游戏板不为空
        if i > 5:  # 前5次测试使用空白板
            num_random_blocks = random.randint(1, 5)
            for _ in range(num_random_blocks):
                piece = random.choice(shapes)
                rotations = random.randint(0, 3)
                for _ in range(rotations):
                    piece = rotate(piece)
                x = random.randint(0, 10 - len(piece[0]))
                y = random.randint(10, 19 - len(piece))
                if check(board, piece, [x, y]):
                    join_matrix(board, piece, [x, y])
        
        # 随机选择一个方块
        piece = random.choice(shapes)
        print(f"测试方块形状 {shapes.index(piece)}:")
        for row in piece:
            print(''.join(['□' if cell == 0 else '■' for cell in row]))
        
        # 预测移动
        try:
            move = ai.predict_move(board, piece)
            if move:
                print(f"预测移动: x={move['x']}, y={move['y']}, 旋转={move['rotation']}")
                
                # 验证移动
                rotated_piece = deepcopy(piece)
                for _ in range(move['rotation']):
                    rotated_piece = rotate(rotated_piece)
                
                # 可视化并获取有效性
                is_valid, image_path = visualize_move(board, piece, move, i, timestamp)
                
                # 更新统计
                stats['total'] += 1
                if is_valid:
                    stats['valid'] += 1
                    print("移动有效 ✓")
                else:
                    stats['invalid'] += 1
                    print("移动无效 ❌")
                
                stats['details'].append({
                    'test_num': i+1,
                    'piece_type': shapes.index(piece),
                    'is_valid': is_valid,
                    'move': move,
                    'image': image_path
                })
            else:
                print("无移动预测")
        except Exception as e:
            print(f"测试 {i+1} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成报告
    generate_report(stats, model_path, timestamp)
    return stats

def generate_report(stats, model_path, timestamp):
    """生成诊断报告"""
    valid_rate = stats['valid'] / stats['total'] * 100 if stats['total'] > 0 else 0
    
    report_path = f"{DIAGNOSTICS_DIR}/{timestamp}_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"===== 俄罗斯方块监督学习系统 - 移动诊断报告 =====\n")
        f.write(f"模型: {model_path}\n")
        f.write(f"诊断时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试总数: {stats['total']}\n")
        f.write(f"有效移动: {stats['valid']} ({valid_rate:.1f}%)\n")
        f.write(f"无效移动: {stats['invalid']} ({100-valid_rate:.1f}%)\n\n")
        
        f.write("移动详情:\n")
        invalid_moves = [d for d in stats['details'] if not d['is_valid']]
        if invalid_moves:
            f.write(f"无效移动清单 (共 {len(invalid_moves)} 个):\n")
            for i, d in enumerate(invalid_moves):
                f.write(f"  {i+1}. 测试 {d['test_num']}: 方块类型 {d['piece_type']}, "
                      f"移动: x={d['move']['x']}, y={d['move']['y']}, 旋转={d['move']['rotation']}\n")
        else:
            f.write("没有发现无效移动!\n")
            
    print(f"\n诊断完成! 报告已保存到: {report_path}")
    
    # 生成结果图表
    try:
        plt.figure(figsize=(8, 6))
        labels = ['有效移动', '无效移动']
        sizes = [stats['valid'], stats['invalid']]
        colors = ['#66b266', '#ff6666']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, shadow=True)
        plt.axis('equal')
        plt.title(f'移动有效性分析 - {os.path.basename(model_path)}')
        pie_chart_path = f"{DIAGNOSTICS_DIR}/{timestamp}_valid_rates.png"
        plt.savefig(pie_chart_path)
        plt.close()
        print(f"统计图表已保存到: {pie_chart_path}")
    except Exception as e:
        print(f"生成图表出错: {e}")

if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="俄罗斯方块模型诊断工具")
    parser.add_argument('-m', '--model', type=str, help='要诊断的模型文件路径')
    parser.add_argument('-n', '--num-tests', type=int, default=50, help='测试次数 (默认: 50)')
    args = parser.parse_args()

    if not args.model:
        print("错误: 请指定要诊断的模型文件路径")
        parser.print_help()
        sys.exit(1)

    diagnose_moves(args.model, args.num_tests)
