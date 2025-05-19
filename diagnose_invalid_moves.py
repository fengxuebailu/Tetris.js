#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 无效移动诊断工具
用于分析和诊断神经网络预测的无效移动问题
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
from copy import deepcopy
import os
import torch
import sys
from Tetris import shapes, rotate, check, join_matrix, clear_rows

# 历史记录保存路径
DIAGNOSTICS_DIR = "move_diagnostics"

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
        # 尝试使用改进的兼容性加载
        try:
            from model_compatibility import load_old_model
            old_model, predict_func, model_type = load_old_model(model_path)
            print(f"成功加载模型: {model_path} (类型: {model_type})")
            
            # 创建一个类似TetrisAI的接口
            class CompatAI:
                def __init__(self, predict_func):
                    self.predict_func = predict_func
                    self.model_type = model_type
                    
                def predict_move(self, board, piece):
                    from tetris_supervised_fixed import TetrisDataCollector
                    collector = TetrisDataCollector()
                    x, rotation = self.predict_func(board, piece, collector.create_state_vector)
                    # 详细记录原始输出
                    print(f"模型原始输出: x={x}, rotation={rotation}")
                    
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
                    
                    # 最终验证并尝试修复
                    if not check(board, rotated_piece, [x, y]):
                        print(f"尝试修复无效移动: x={x}, y={y}, rotation={rotation}")
                        # 尝试多种调整策略
                        # 1. 尝试调整x
                        for test_x in range(max(-2, x-2), min(len(board[0])+2, x+3)):
                            if check(board, rotated_piece, [test_x, y]):
                                x = test_x
                                print(f"通过调整x={test_x}修复了无效移动")
                                break
                        # 2. 如果调整x无效，尝试使用不同旋转
                        if not check(board, rotated_piece, [x, y]):
                            for test_rot in range(4):
                                if test_rot == rotation:
                                    continue
                                test_piece = deepcopy(piece)
                                for _ in range(test_rot):
                                    test_piece = rotate(test_piece)
                                # 重新计算下落位置
                                test_y = 0
                                while test_y < len(board) and check(board, test_piece, [x, test_y+1]):
                                    test_y += 1
                                if check(board, test_piece, [x, test_y]):
                                    y = test_y
                                    rotation = test_rot
                                    rotated_piece = test_piece
                                    print(f"通过调整rotation={test_rot}修复了无效移动")
                                    break
                    
                    return {'x': x, 'y': y, 'rotation': rotation}
            
            ai = CompatAI(predict_func)
            print(f"成功使用兼容模式加载模型: {model_path}")
            return ai, model_type
        except Exception as e:
            # 如果兼容加载失败，尝试直接加载
            print(f"兼容模式加载失败: {e}，尝试直接加载...")
            
            # 先尝试增强版AI
            try:
                from tetris_supervised_fixed import TetrisAI
                ai = TetrisAI(model_path)
                print(f"成功加载AI模型 (新架构): {model_path}")
                return ai, "new"
            except Exception as e2:
                print(f"使用新架构加载失败: {e2}，尝试旧架构...")
                # 尝试旧版AI
                try:
                    from tetris_supervised import TetrisAI as OldTetrisAI
                    ai = OldTetrisAI(model_path)
                    print(f"成功加载AI模型 (旧架构): {model_path}")
                    return ai, "old"
                except Exception as e3:
                    print(f"使用旧架构加载也失败: {e3}")
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
        if piece_array.shape[0] > 0 and piece_array.shape[1] > 0:
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
        new_board = [row[:] for row in board_copy]
        if is_valid:
            join_matrix(new_board, rotated_piece, [move['x'], move['y']])
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
        
        ax2.imshow(new_board, cmap='binary', alpha=0.8)
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
                
                is_valid = check(board, rotated_piece, [move['x'], move['y']])
                
                # 可视化
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
        
def display_help():
    """显示帮助信息"""
    print("俄罗斯方块监督学习系统 - 无效移动诊断工具")
    print("用法: python diagnose_invalid_moves.py [选项]")
    print("\n选项:")
    print("  -h, --help             显示此帮助信息")
    print("  -m, --model MODEL      指定要诊断的模型路径")
    print("  -n, --num-tests NUM    指定测试次数 (默认: 50)")
    print("  -c, --compare          比较所有可用模型")

def compare_all_models(num_tests=20):
    """比较所有可用模型的有效移动率"""
    # 查找所有模型文件
    model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
    
    if not model_files:
        print("错误: 找不到任何模型文件")
        return
        
    print(f"找到 {len(model_files)} 个模型文件:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    # 诊断每个模型
    results = {}
    for model_file in model_files:
        print(f"\n===== 诊断模型: {model_file} =====")
        stats = diagnose_moves(model_file, num_tests)
        if stats:
            valid_rate = stats['valid'] / stats['total'] * 100 if stats['total'] > 0 else 0
            results[model_file] = {
                'valid_rate': valid_rate,
                'valid': stats['valid'],
                'invalid': stats['invalid'],
                'total': stats['total']
            }
    
    # 生成比较报告
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = f"{DIAGNOSTICS_DIR}/{timestamp}_models_comparison.txt"
    with open(report_path, 'w') as f:
        f.write("===== 俄罗斯方块监督学习系统 - 模型有效移动率比较 =====\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"每个模型测试次数: {num_tests}\n\n")
        
        f.write(f"{'模型':<30} {'有效率':<10} {'有效':<8} {'无效':<8} {'总数':<8}\n")
        f.write("-" * 70 + "\n")
        
        # 按有效率排序
        sorted_models = sorted(results.items(), key=lambda x: x[1]['valid_rate'], reverse=True)
        for model_name, stats in sorted_models:
            f.write(f"{model_name:<30} {stats['valid_rate']:>8.1f}% {stats['valid']:>8} {stats['invalid']:>8} {stats['total']:>8}\n")
    
    print(f"\n比较完成! 报告已保存到: {report_path}")
    
    # 生成比较图表
    try:
        plt.figure(figsize=(12, 6))
        model_names = [os.path.basename(m) for m, _ in sorted_models]
        valid_rates = [s['valid_rate'] for _, s in sorted_models]
        
        plt.bar(model_names, valid_rates, color='#5599ff')
        plt.axhline(y=90, color='r', linestyle='--', label='90%阈值')
        plt.xticks(rotation=45, ha='right')
        plt.title('各模型有效移动率对比')
        plt.ylabel('有效移动率 (%)')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        comparison_path = f"{DIAGNOSTICS_DIR}/{timestamp}_models_comparison.png"
        plt.savefig(comparison_path)
        plt.close()
        print(f"比较图表已保存到: {comparison_path}")
    except Exception as e:
        print(f"生成图表出错: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        display_help()
    elif sys.argv[1] in ['-c', '--compare']:
        num_tests = 20
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            num_tests = int(sys.argv[2])
        compare_all_models(num_tests)
    elif sys.argv[1] in ['-m', '--model']:
        if len(sys.argv) < 3:
            print("错误: 未指定模型路径")
            display_help()
        else:
            model_path = sys.argv[2]
            num_tests = 50
            if len(sys.argv) > 4 and sys.argv[3] in ['-n', '--num-tests'] and sys.argv[4].isdigit():
                num_tests = int(sys.argv[4])
            diagnose_moves(model_path, num_tests)
    else:
        # 默认诊断所有模型
        compare_all_models(20)
