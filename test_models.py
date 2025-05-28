#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare two trained Tetris models.
"""
import argparse
import sys
import time
import random
import numpy as np
import torch
import os

# 获取脚本的绝对路径
ABS_PATH_OF_SCRIPT = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = ABS_PATH_OF_SCRIPT # Assuming test_models.py is in the project root

# 将项目根目录添加到 sys.path，以便可以执行 supervised_learning.core.module
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 尝试导入必要的模块
try:
    from supervised_learning.core.tetris_supervised_fixed import TetrisAI, TetrisNet
    from supervised_learning.core.Tetris import shapes, rotate, check, join_matrix, clear_rows
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print("sys.path:")
    for p in sys.path:
        print(f"  - {p}")
    print("\nPlease ensure that the script is run from the root of the 'Tetris.js' project directory.")
    sys.exit(1)


def simulate_game(ai_model, num_games=10, max_moves_per_game=500, display=False):
    """
    模拟指定数量的游戏，并返回平均得分和行数。

    参数:
    - ai_model: TetrisAI 实例。
    - num_games: 要模拟的游戏数量。
    - max_moves_per_game: 每局游戏的最大移动次数。
    - display: 是否显示游戏过程（简化版）。

    返回:
    - (平均得分, 平均消除行数)
    """
    total_scores = []
    total_lines_cleared = []

    for i_game in range(num_games):
        board = [[0 for _ in range(10)] for _ in range(20)]
        score = 0
        lines_cleared_this_game = 0
        game_over = False
        moves_count = 0
        current_piece_shape = random.choice(shapes)


        if display:
            print(f"\n--- Game {i_game + 1} ---")

        while not game_over and moves_count < max_moves_per_game:
            # AI 预测移动，传递实际的方块形状
            predicted_move = ai_model.predict_move(board, current_piece_shape)
            
            x = predicted_move['x']
            rotation = predicted_move['rotation']
            
            # 应用旋转
            rotated_piece = current_piece_shape
            for _ in range(rotation):
                rotated_piece = rotate(rotated_piece)
            
            # 找到下落位置 y
            y = 0
            if check(board, rotated_piece, (x, 0)):
                while check(board, rotated_piece, (x, y + 1)):
                    y += 1
            else:
                game_over = True
                if display: print(f"Game Over - Cannot place new piece at initial position. x={x}, rot={rotation}, piece={current_piece_shape}")
                break

            # 放置方块
            join_matrix(board, rotated_piece, (x, y))
            
            # 清除行并计分
            new_board, num_cleared = clear_rows(board)
            board = new_board
            
            if num_cleared > 0:
                lines_cleared_this_game += num_cleared
                if num_cleared == 1:
                    score += 40
                elif num_cleared == 2:
                    score += 100
                elif num_cleared == 3:
                    score += 300
                elif num_cleared >= 4:
                    score += 1200
            
            moves_count += 1
            # 获取下一个方块
            current_piece_shape = random.choice(shapes)

            if display and moves_count % 50 == 0:
                print(f"Move {moves_count}, Score: {score}, Lines: {lines_cleared_this_game}")
        
        if display:
            print(f"Game {i_game + 1} ended. Score: {score}, Lines Cleared: {lines_cleared_this_game}, Moves: {moves_count}")

        total_scores.append(score)
        total_lines_cleared.append(lines_cleared_this_game)

    avg_score = np.mean(total_scores) if total_scores else 0
    avg_lines = np.mean(total_lines_cleared) if total_lines_cleared else 0
    
    return avg_score, avg_lines

def resolve_model_path(path_arg, base_dir):
    """
    解析模型路径。
    尝试将 path_arg 视为:
    1. 绝对路径
    2. 相对于 base_dir 的路径
    3. 相对于 base_dir/supervised_learning/models/ 的路径 (如果 path_arg 只是文件名)
    4. 相对于 base_dir/path_arg (如果 path_arg 包含 'supervised_learning/models/')
    """
    # 1. 绝对路径
    if os.path.isabs(path_arg) and os.path.exists(path_arg):
        return path_arg

    # 2. 相对于 base_dir (通常是项目根目录)
    path_from_base = os.path.join(base_dir, path_arg)
    if os.path.exists(path_from_base):
        return path_from_base
        
    # 3. 如果 path_arg 只是文件名，则在 supervised_learning/models/ 中查找
    if not os.path.dirname(path_arg): # 如果 path_arg 是文件名
        path_in_models_dir = os.path.join(base_dir, "supervised_learning", "models", path_arg)
        if os.path.exists(path_in_models_dir):
            return path_in_models_dir
            
    # 4. 如果 path_arg 已经是类似 "supervised_learning/models/model.pth" 的形式
    # (这种情况应该被第2点覆盖，但为了明确性可以保留)
    # path_from_base (已在上面计算)

    print(f"Error: Model file not found for argument: {path_arg}")
    print(f"  Tried absolute: {path_arg}")
    print(f"  Tried relative to project root ({base_dir}): {path_from_base}")
    if not os.path.dirname(path_arg):
         print(f"  Tried in default models folder: {os.path.join(base_dir, '''supervised_learning''', '''models''', path_arg)}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Compare two Tetris AI models.")
    parser.add_argument("model_path1", type=str, help="Path to the first model file (.pth).")
    parser.add_argument("model_path2", type=str, help="Path to the second model file (.pth).")
    parser.add_argument("--games", type=int, default=10, help="Number of games to simulate for each model.") # Reduced default for faster testing
    parser.add_argument("--max_moves", type=int, default=300, help="Maximum moves per game.") # Reduced default
    parser.add_argument("--display", action="store_true", help="Display simplified game progress.")
    
    args = parser.parse_args()

    print(f"Comparing Model 1 ({os.path.basename(args.model_path1)}) vs Model 2 ({os.path.basename(args.model_path2)})")
    print(f"Simulating {args.games} games for each model, with up to {args.max_moves} moves per game.")

    model_path1_resolved = resolve_model_path(args.model_path1, PROJECT_ROOT)
    model_path2_resolved = resolve_model_path(args.model_path2, PROJECT_ROOT)

    print(f"Loading Model 1 from: {model_path1_resolved}")
    print(f"Loading Model 2 from: {model_path2_resolved}")

    try:
        ai1 = TetrisAI(model_path=model_path1_resolved)
    except Exception as e:
        print(f"Error loading Model 1 ({model_path1_resolved}): {e}")
        sys.exit(1)
        
    try:
        ai2 = TetrisAI(model_path=model_path2_resolved)
    except Exception as e:
        print(f"Error loading Model 2 ({model_path2_resolved}): {e}")
        sys.exit(1)

    print("\n--- Simulating Model 1 ---")
    start_time = time.time()
    avg_score1, avg_lines1 = simulate_game(ai1, args.games, args.max_moves, args.display)
    time1 = time.time() - start_time
    print(f"Model 1 - Avg Score: {avg_score1:.2f}, Avg Lines: {avg_lines1:.2f}, Time: {time1:.2f}s")

    print("\n--- Simulating Model 2 ---")
    start_time = time.time()
    avg_score2, avg_lines2 = simulate_game(ai2, args.games, args.max_moves, args.display)
    time2 = time.time() - start_time
    print(f"Model 2 - Avg Score: {avg_score2:.2f}, Avg Lines: {avg_lines2:.2f}, Time: {time2:.2f}s")

    print("\n--- Comparison Summary ---")
    print(f"Metric        | Model 1 ({os.path.basename(args.model_path1)}) | Model 2 ({os.path.basename(args.model_path2)}) | Difference (M2 - M1)")
    print("----------------|---------------------------|---------------------------|----------------------")
    print(f"Avg Score     | {avg_score1:<25.2f} | {avg_score2:<25.2f} | {avg_score2 - avg_score1:^+20.2f}")
    print(f"Avg Lines     | {avg_lines1:<25.2f} | {avg_lines2:<25.2f} | {avg_lines2 - avg_lines1:^+20.2f}")
    print(f"Sim Time (s)  | {time1:<25.2f} | {time2:<25.2f} | {time2 - time1:^+20.2f}")

    if avg_score1 == avg_score2:
        print("\nOverall: Models performed similarly in terms of score.")
    elif avg_score2 > avg_score1:
        print(f"\nOverall: Model 2 ({os.path.basename(args.model_path2)}) performed better on average score.")
    else:
        print(f"\nOverall: Model 1 ({os.path.basename(args.model_path1)}) performed better on average score.")

if __name__ == "__main__":
    main()
