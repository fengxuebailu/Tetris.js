#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统
使用神经网络从专家系统收集的数据中学习
"""

# Standard library imports
import sys
import os
import random
import time
import json
from copy import deepcopy
import traceback

# Third-party library imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # 新添加的导入
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Local application/library specific imports
# 从Tetris.py导入游戏核心功能，不导入curses相关的内容
from Tetris import shapes, rotate, check, join_matrix, clear_rows, get_height, count_holes, get_bumpiness

# Helper function for printing board (used in mode 3)
def print_board(board_state):
    """Helper function to print the board (text-based)."""
    # Simple text-based print for a board
    print("-" * (len(board_state[0]) * 2 + 1))
    for row in board_state:
        print("|" + "|".join(["X" if cell else " " for cell in row]) + "|")
    print("-" * (len(board_state[0]) * 2 + 1))

# 封装和处理训练数据
class TetrisDataset(Dataset):
    """用于存储和加载俄罗斯方块游戏状态和对应的最佳移动"""
    def __init__(self, game_states, moves):
        self.game_states = torch.FloatTensor(game_states)  # 游戏状态张量
        self.moves = torch.FloatTensor(moves)  # 对应的最佳移动张量
        print(f"创建数据集: {len(self.game_states)} 个样本")
        print(f"状态张量形状: {self.game_states.shape}")
        print(f"移动张量形状: {self.moves.shape}")

    def __len__(self):
        return len(self.game_states)

    def __getitem__(self, idx):
        return self.game_states[idx], self.moves[idx]

    @staticmethod
    def save_to_file(game_states, moves, filename='tetris_training_data.npz'):
        """将收集的数据保存到文件"""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_data")
        file_path = os.path.join(data_dir, filename)
        np.savez(file_path, 
                 states=game_states, 
                 moves=moves)
        print(f"训练数据已保存到 {file_path}")

    @staticmethod
    def load_from_file(filename='tetris_training_data.npz'):
        """从文件加载训练数据"""
        try:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_data")
            file_path = os.path.join(data_dir, filename)
            data = np.load(file_path)
            states_data = data['states']
            moves_data = data['moves']
            
            # Safely get lengths
            states_len = len(states_data) if hasattr(states_data, '__len__') else 0
            moves_len = len(moves_data) if hasattr(moves_data, '__len__') else 0
            
            print(f"加载数据: {states_len} 个状态, {moves_len} 个移动")
            # Ensure that if one is None or empty, both are returned as such or handled appropriately
            if states_len == 0 or moves_len == 0 or states_len != moves_len:
                print("警告: 加载的数据无效或不完整。")
                return None, None
            return states_data, moves_data
        except FileNotFoundError:
            print(f"错误: 训练数据文件 {file_path} 未找到。")
            return None, None
        except Exception as e:
            print(f"加载训练数据出错: {e}")
            return None, None

# 神经网络的结构
class TetrisNet(nn.Module):
    def __init__(self):
        super(TetrisNet, self).__init__()
        # 输入特征: 游戏板状态(200) + 高度(1) + 空洞(1) + 凹凸度(1) + 当前方块(16) = 219
        
        # 使用CNN处理游戏板状态
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1))  # 输出大小为 64x1x1
        )
        
        # 处理额外特征
        self.extra_features = nn.Sequential(
            nn.Linear(3, 32),  # 高度、空洞、凹凸度
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1)
        )
        
        # 处理当前方块
        self.piece_features = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32)
        )
        
        # 合并所有特征的决策网络
        self.decision_network = nn.Sequential(
            nn.Linear(64 + 32 + 32, 128),  # 64(conv) + 32(extra) + 32(piece) = 128
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 2)  # 输出: x位置, 旋转角度
        )
        
    def forward(self, x):
        # 分离不同类型的输入
        board_input = x[:, :200]  # 游戏板状态 (200)
        extra_input = x[:, 200:203]  # 额外特征 (3)
        piece_input = x[:, 203:]  # 方块特征 (16)
        
        # 处理游戏板 - 重塑为 2D 并添加通道维度
        board_feats = board_input.view(-1, 1, 20, 10)  # (batch, channel, height, width)
        board_feats = self.conv_layers(board_feats)  # CNN处理
        board_feats = board_feats.view(-1, 64)  # 展平为 (batch, 64)
        
        # 处理额外特征和方块特征
        extra_feats = self.extra_features(extra_input)
        piece_feats = self.piece_features(piece_input)
        
        # 合并所有特征
        combined = torch.cat([board_feats, extra_feats, piece_feats], dim=1)
        
        # 生成最终决策
        return self.decision_network(combined)

def load_weights():
    """加载或创建权重"""
    # Try to load from project root first, then from core directory as a fallback for old behavior
    # This assumes the script might be run from different working directories.
    # Best practice is to make paths absolute or relative to the script file.
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_canary_file = 'best_weights_evolved.json' # The file we are looking for
    
    # Path relative to script: ../../best_weights_evolved.json (core -> supervised_learning -> project_root)
    path_in_project_root = os.path.join(script_dir, '..', '..', project_root_canary_file)
    
    # Path relative to current working dir (if script is run from project root)
    path_in_cwd = project_root_canary_file

    paths_to_try = [
        path_in_project_root,
        path_in_cwd,
    ]

    for weights_path in paths_to_try:
        try:
            # print(f"尝试加载权重文件: {os.path.abspath(weights_path)}") # Debug print
            with open(weights_path, 'r') as f:
                weights_data = json.load(f)
                print(f"成功从 {os.path.abspath(weights_path)} 加载权重。")
                # 确保所有值都是浮点数
                return {k: float(v) for k, v in weights_data.items()}
        except FileNotFoundError:
            # print(f"权重文件未找到于: {os.path.abspath(weights_path)}") # Debug print
            continue
        except Exception as e:
            print(f"加载权重时发生错误 {os.path.abspath(weights_path)}: {e}")
            continue # Try next path or fall through to default

    print("所有尝试路径均未找到权重文件 (best_weights_evolved.json)，使用默认权重。")
    return {
        'cleared_lines': 160.0,
        'holes': -50.0,
        'bumpiness': -20.0,
        'height': -30.0  # Ensure 'height' is included if used by heuristic
    }

class TetrisDataCollector:
    def __init__(self, num_games=100, max_moves=200, timeout=60):
        self.num_games = num_games
        self.max_moves = max_moves  # 每个游戏的最大步数
        self.timeout = timeout  # 每个游戏的最大运行时间(秒)
        self.game_states = []
        self.optimal_moves = []
        
    def collect_data(self):
        """使用启发式方法收集游戏数据"""
        print("开始收集数据...")
        try:
            # 使用之前进化算法得到的最佳权重
            weights = load_weights()
            print("成功加载权重:", weights)
            
            total_moves = 0
            start_time = time.time()
            
            for game in range(self.num_games):
                print(f"收集游戏数据 {game + 1}/{self.num_games}")
                board = [[0 for _ in range(10)] for _ in range(20)]
                moves_count = 0
                game_start_time = time.time()
                
                # 限制每个游戏的时间
                while moves_count < self.max_moves and (time.time() - game_start_time) < self.timeout : # Added per-game timeout check
                    
                    # 随机选择一个方块
                    current_piece = random.choice(shapes)
                    
                    # 使用简化版查找最佳移动
                    move_data = self.find_best_move_optimized(board, current_piece, weights)
                    
                    if move_data is None:
                        print(f"警告: 游戏 {game + 1} 在第 {moves_count} 步找不到最佳移动，可能提前结束此局。")
                        break # 结束当前游戏
                    
                    # 保存游戏状态和最佳移动
                    state_vector = self.create_state_vector(board, current_piece)
                    self.game_states.append(state_vector)
                    self.optimal_moves.append([
                        move_data['x'],
                        move_data['rotation']
                    ])
                    
                    # 执行移动
                    rotated_piece = deepcopy(current_piece)
                    for _ in range(move_data['rotation']):
                        rotated_piece = rotate(rotated_piece) # Ensure piece is actually rotated
                    
                    # 更新游戏板
                    join_matrix(board, rotated_piece, [move_data['x'], move_data['y']])
                    # clear_rows in Tetris.py returns (new_board, cleared_lines_count)
                    new_board_state, cleared_this_step = clear_rows(board) 
                    board = new_board_state # Update board
                    moves_count += 1
                    total_moves += 1
                    
                    # 每20步打印进度
                    if moves_count % 20 == 0:
                        print(f"  游戏 {game+1}, 步数 {moves_count}/{self.max_moves}")
                        
                print(f"游戏 {game + 1} 完成，收集了 {moves_count} 步，耗时: {time.time() - game_start_time:.1f}秒")
                
                # 检查总体超时
                if time.time() - start_time > self.timeout * self.num_games * 0.75: # Adjusted overall timeout logic
                    print("数据收集总体时间过长，提前结束")
                    break

            # 转换为numpy数组并设定数据类型
            print(f"完成数据收集，共收集 {total_moves} 步，{len(self.game_states)} 个状态")
            print(f"总耗时: {time.time() - start_time:.1f}秒")
            
            game_states_array = np.array(self.game_states, dtype=np.float32)
            optimal_moves_array = np.array(self.optimal_moves, dtype=np.float32)
            
            # 保存训练数据
            TetrisDataset.save_to_file(game_states_array, optimal_moves_array)
            
            return game_states_array, optimal_moves_array
        except Exception as e:
            print(f"收集数据时发生错误: {str(e)}")
            traceback.print_exc()
            
            # 即使发生错误，也尝试返回已收集的数据
            if len(self.game_states) > 0 and len(self.optimal_moves) > 0:
                return np.array(self.game_states, dtype=np.float32), np.array(self.optimal_moves, dtype=np.float32)
            # raise # Re-raising might be too disruptive if some data was collected.
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32) # Return empty arrays if nothing useful

    def create_state_vector(self, board, piece):
        """创建游戏状态向量，确保方块向量大小一致"""
        # 展平游戏板 (20x10 = 200)
        board_vector = np.array(board, dtype=np.float32).flatten()

        # 计算额外的启发式特征
        height_val = np.array([get_height(board)], dtype=np.float32)
        holes_val = np.array([count_holes(board)], dtype=np.float32)
        bumpiness_val = np.array([get_bumpiness(board)], dtype=np.float32)
        
        # 将piece转换为4x4矩阵
        piece_matrix = np.zeros((4, 4), dtype=np.float32)
        # Ensure piece is a numpy array for shape attribute
        if not isinstance(piece, np.ndarray):
            piece_array = np.array(piece)
        else:
            piece_array = piece
            
        h, w = piece_array.shape
        # 将piece放在4x4矩阵的中心
        start_h = (4 - h) // 2
        start_w = (4 - w) // 2
        piece_matrix[start_h:start_h+h, start_w:start_w+w] = piece_array
        piece_vector = piece_matrix.flatten()  # 4x4 = 16
        
        # 合并状态 (200 + 1 + 1 + 1 + 16 = 219)
        return np.concatenate([board_vector, height_val, holes_val, bumpiness_val, piece_vector])
        
    def find_best_move_optimized(self, board, piece, weights):
        """优化版本的最佳移动查找算法，添加了超时检测和优化"""
        best_score = float('-inf')
        best_move = None
        start_time = time.time()
        
        # 优化：根据方块形状决定需要测试的旋转次数
        # 长条形状(I): 只需要2次旋转
        # 方形(O): 只需要1次旋转
        # 其他: 需要4次旋转
        shape_type = "unknown"
        if len(piece) == 1 and len(piece[0]) == 4:  # I形
            max_rotations = 2
            shape_type = "I"
        elif len(piece) == 2 and len(piece[0]) == 2:  # O形
            max_rotations = 1
            shape_type = "O"
        else:  # 其他形状
            max_rotations = 4
            shape_type = "other"
        
        # 优化：根据方块形状设置合理的x轴范围
        # width_orig = len(piece[0]) # Use width of rotated piece later
        min_x_orig = 0
        # max_x_orig = len(board[0]) - width_orig

        # if width_orig == 4:  # I形水平方向
        #     min_x_orig = -1
        #     max_x_orig = len(board[0]) - 2
        
        max_eval_time = 0.5
        
        rotated_pieces = []
        current = deepcopy(piece)
        for r_idx in range(max_rotations):
            rotated_pieces.append(current)
            if r_idx < max_rotations -1 : # Avoid rotating one too many times if max_rotations is 1
                 current = rotate(current)
            
        evaluated = 0
        for rotation_idx, current_piece_rotated in enumerate(rotated_pieces):
            width_rotated = len(current_piece_rotated[0])
            # height_rotated = len(current_piece_rotated) # unused
            
            # Dynamic x_range based on rotated piece
            # width_rotated is len(current_piece_rotated[0])
            # current_min_x = -width_rotated + 1 

            # Align with the x-range from play_with_evolved_weights.py for consistency
            # The range in play_with_evolved_weights.py is:
            # range(-len(current_rotated_piece[0]) + 1, len(board[0]) + 2)
            # which translates to:
            # range(-width_rotated + 1, len(board[0]) + 2)

            for x_candidate in range(-width_rotated + 1, len(board[0]) + 2):
                if time.time() - start_time > max_eval_time:
                    # print(f"评估超时，已评估 {evaluated} 个位置") # Can be noisy
                    if best_move is not None: return best_move # Return best found so far
                # print(f"Checking position x={x_candidate}, rotation={rotation_idx}")
                if not check(board, current_piece_rotated, [x_candidate, 0]):
                    continue
                
                evaluated += 1
                y_final = 0
                while y_final < len(board) -1 and check(board, current_piece_rotated, [x_candidate, y_final + 1]):
                    y_final += 1
                
                if not check(board, current_piece_rotated, [x_candidate, y_final]): # Double check final position
                    continue

                temp_board = [row[:] for row in board]
                join_matrix(temp_board, current_piece_rotated, [x_candidate, y_final])
                
                new_board_after_clear, cleared_count = clear_rows(temp_board)
                
                current_score = (weights['cleared_lines'] * cleared_count +
                                 weights['holes'] * count_holes(new_board_after_clear) +
                                 weights['bumpiness'] * get_bumpiness(new_board_after_clear) +
                                 weights['height'] * get_height(new_board_after_clear))
                
                if current_score > best_score:
                    best_score = current_score
                    best_move = {
                        'x': x_candidate,
                        'y': y_final,
                        'rotation': rotation_idx # Use rotation_idx corresponding to current_piece_rotated
                    }
        
        if best_move is None:
            print("未找到有效移动，尝试第二轮搜索...")
            for r_idx, current_piece_rotated_fallback in enumerate(rotated_pieces):
                width_fallback = len(current_piece_rotated_fallback[0])
                for x_fb in range(-width_fallback +1, len(board[0]) - width_fallback + 1):
                    if check(board, current_piece_rotated_fallback, [x_fb, 0]):
                        y_fb = 0
                        while y_fb < len(board) -1 and check(board, current_piece_rotated_fallback, [x_fb, y_fb + 1]):
                            y_fb += 1
                        
                        if check(board, current_piece_rotated_fallback, [x_fb, y_fb]): # Ensure final placement is valid
                            print(f"第二轮搜索成功: x={x_fb}, y={y_fb}, 旋转={r_idx}")
                            return {
                                'x': x_fb,
                                'y': y_fb,
                                'rotation': r_idx
                            }
            print("警告: 第二轮搜索也未能找到任何有效移动。返回 None。")
            return None # Ensure None is returned if fallback also fails
            


        return best_move
            


    


class WeightedTetrisLoss(nn.Module):
    """加权俄罗斯方块损失函数"""
    def __init__(self, x_weight=1.0, rotation_weight=2.0):
        super().__init__()
        self.x_weight = x_weight
        self.rotation_weight = rotation_weight
    
    def forward(self, outputs, targets):
        # 分别计算x坐标和旋转的损失
        x_loss = F.mse_loss(outputs[:, 0], targets[:, 0])
        rotation_loss = F.mse_loss(outputs[:, 1], targets[:, 1])
        
        # 加权组合
        total_loss = self.x_weight * x_loss + self.rotation_weight * rotation_loss
        return total_loss, x_loss, rotation_loss

def train_network(game_states, moves, num_epochs=200, batch_size=32, learning_rate=0.001, patience=30):
    # 创建数据集
    dataset = TetrisDataset(game_states, moves)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    model = TetrisNet()
    # 使用新的损失函数替代原来的 MSELoss
    criterion = WeightedTetrisLoss(x_weight=1.0, rotation_weight=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 学习率调度器 - 当验证损失不再下降时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,      # 每次降低一半
        patience=10      # 增加等待轮数
    )

    # 添加手动打印学习率变化的代码
    last_lr = learning_rate
    def log_lr_change(new_lr):
        nonlocal last_lr
        if new_lr != last_lr:
            print(f'学习率从 {last_lr:.6f} 调整为 {new_lr:.6f}')
            last_lr = new_lr

    print("开始训练模型...")
    print(f"训练数据: {train_size} 个样本, 验证数据: {val_size} 个样本")
    print(f"批次大小: {batch_size}, 最大轮数: {num_epochs}")

    # 早停策略变量
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    # 用于存储训练历史的列表
    train_loss_history = []
    val_loss_history = []
    learning_rates_history = []

    # 训练循环
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_x_loss = 0
        train_rot_loss = 0
        printed_sample_this_epoch = False  # 在每个epoch开始时初始化标志

        for batch_idx, (batch_states, batch_moves) in enumerate(train_loader):
            # 前向传播
            outputs = model(batch_states)
            loss, x_loss, rot_loss = criterion(outputs, batch_moves)

            # 打印批次损失 (每20个批次)
            if batch_idx % 20 == 0:
                print(f"    Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Batch Loss: {loss.item():.4f}")

            # 采样和打印数据 (每个epoch的第一个批次)
            if not printed_sample_this_epoch and batch_states.size(0) > 0:
                print(f"--- Sample Data - Epoch {epoch+1} ---")
                sample_input_tensor = batch_states[0]
                sample_target_tensor = batch_moves[0]
                
                # Detach and convert to numpy for printing
                sample_input_np = sample_input_tensor.cpu().numpy()
                sample_target_np = sample_target_tensor.cpu().numpy()
                
                with torch.no_grad(): # Ensure no gradients are computed for this sample forward pass
                    model.eval() # Set model to eval mode for consistent output
                    # Unsqueeze to add batch dimension for single sample prediction
                    sample_output_tensor = model(sample_input_tensor.unsqueeze(0)).squeeze(0) 
                    model.train() # Set model back to train mode
                
                sample_output_np = sample_output_tensor.cpu().numpy()
                
                print(f"  Sample Input (first 20 features): {sample_input_np[:20]}...")
                print(f"  Expert Target: x={sample_target_np[0]:.2f}, rotation={sample_target_np[1]:.2f}")
                print(f"  Model Output:  x={sample_output_np[0]:.2f}, rotation={sample_output_np[1]:.2f}")
                print(f"--- End Sample Data ---")
                printed_sample_this_epoch = True

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积损失
            train_loss += loss.item()
            train_x_loss += x_loss.item()
            train_rot_loss += rot_loss.item()

            # 每20个批次打印一次详细信息
            if batch_idx % 20 == 0:
                print(f"\nBatch {batch_idx}:")
                print(f"总损失: {loss.item():.4f}")
                print(f"X坐标损失: {x_loss.item():.4f}")
                print(f"旋转损失: {rot_loss.item():.4f}")
                
                # 打印一个样本的具体预测值
                with torch.no_grad():
                    print("\n样本预测:")
                    print(f"预测值: X={outputs[0,0].item():.2f}, 旋转={outputs[0,1].item():.2f}")
                    print(f"目标值: X={batch_moves[0,0].item():.2f}, 旋转={batch_moves[0,1].item():.2f}")

        # 验证阶段
        model.eval()
        val_loss = 0
        val_x_loss = 0
        val_rot_loss = 0
        with torch.no_grad():
            for batch_states, batch_moves in val_loader:
                outputs = model(batch_states)
                total_loss, x_loss, rot_loss = criterion(outputs, batch_moves)  # 解包返回的损失元组
                val_loss += total_loss.item()  # 使用 total_loss
                val_x_loss += x_loss.item()
                val_rot_loss += rot_loss.item()
    
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        # 检查是否有改进
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            epochs_no_improve = 0
            save_model(model, 'tetris_model_best.pth')
        else:
            epochs_no_improve += 1
        
        # 每10轮保存一次模型
        if (epoch + 1) % 10 == 0:
            save_model(model, f'tetris_model_epoch_{epoch+1}.pth')
        
        # 早停检查
        if epochs_no_improve >= patience:
            print(f"早停: {patience}轮内验证损失没有改善")
            break
    
    # 循环结束后的代码
    # 恢复最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
    
    total_time = time.time() - start_time
    print(f"训练完成！总耗时: {total_time:.2f}秒，最佳验证损失: {best_val_loss:.4f}")

    # 保存训练历史
    history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "learning_rates": learning_rates_history
    }
    # 确保 training_logs 目录存在
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    history_file_path = os.path.join(logs_dir, "training_history.json")
    try:
        with open(history_file_path, 'w') as f:
            json.dump(history, f)
        print(f"训练历史已保存到 {history_file_path}")
    except Exception as e:
        print(f"保存训练历史失败: {e}")

    return model

def save_model(model, filename='tetris_model.pth'):
    """保存训练好的模型"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    file_path = os.path.join(models_dir, filename)
    torch.save(model.state_dict(), file_path)
    print(f"模型已保存到 {file_path}")

class TetrisAI:
    """使用训练好的模型来玩Tetris"""
    def __init__(self, model_path='tetris_model.pth'):
        self.model = TetrisNet()
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # 设置为评估模式
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise

    def predict_move(self, board, piece):
        """预测最佳移动"""
        # 将游戏状态转换为模型输入格式
        state_vector = self.create_state_vector(board, piece)
        # 先转换为numpy数组再转换为张量，避免性能警告
        state_tensor = torch.FloatTensor(np.array([state_vector], dtype=np.float32))
        
        # 使用模型预测
        with torch.no_grad():
            output = self.model(state_tensor)
        
        # 输出为 [x位置, 旋转] 
        predicted_x = int(round(output[0][0].item()))
        predicted_rotation = int(round(output[0][1].item())) % 4  # 确保在0-3范围内
        
        # 计算最终位置并处理特殊情况
        return self._compute_valid_move(board, piece, predicted_x, predicted_rotation) # Start with attempt 0
    
    def _compute_valid_move(self, board, piece, predicted_x, predicted_rotation, attempt=0):
        """
        计算有效的移动位置，尝试修正无效移动。
        Args:
            board: 当前游戏板状态。
            piece: 当前方块的原始形状。
            predicted_x: 模型预测的x位置。
            predicted_rotation: 模型预测的旋转次数。
            attempt: 当前尝试修正的次数。
        Returns:
            一个包含 {'x', 'y', 'rotation'} 的字典，表示有效移动。
        """
        if attempt >= 4: # Max attempts for a single predict_move call.
            print(f"警告: _compute_valid_move 达到最大尝试次数 ({attempt}). 返回计算后的默认安全移动。")
            # Try a very simple default: original piece, rotation 0, centered x, dropped.
            default_rot = 0
            default_piece_rotated = deepcopy(piece) # piece is original, rotation 0
            
            default_width = len(default_piece_rotated[0]) if default_piece_rotated and len(default_piece_rotated) > 0 and len(default_piece_rotated[0]) > 0 else 1
            default_x = len(board[0]) // 2 - default_width // 2

            if check(board, default_piece_rotated, [default_x, 0]):
                default_y = 0
                while default_y < len(board) - 1 and check(board, default_piece_rotated, [default_x, default_y + 1]):
                    default_y += 1
                if check(board, default_piece_rotated, [default_x, default_y]): # Final check for the dropped default
                    print(f"最终默认移动: x={default_x}, y={default_y}, rot={default_rot}")
                    return {'x': default_x, 'y': default_y, 'rotation': default_rot}

            print(f"警告: 无法放置基本默认移动。返回绝对默认 x={default_x},y=0,rot={default_rot}。游戏可能即将结束。")
            return {'x': default_x, 'y': 0, 'rotation': default_rot} # Absolute fallback

        current_x = predicted_x
        current_rotation = predicted_rotation

        if attempt == 1: # Try centering the piece with predicted_rotation
            temp_rotated_piece_for_width = deepcopy(piece)
            for _ in range(predicted_rotation): # Use original predicted_rotation for this attempt
                temp_rotated_piece_for_width = rotate(temp_rotated_piece_for_width)
            piece_width_for_centering = len(temp_rotated_piece_for_width[0]) if temp_rotated_piece_for_width and len(temp_rotated_piece_for_width) > 0 and len(temp_rotated_piece_for_width[0]) > 0 else 1
            current_x = len(board[0]) // 2 - piece_width_for_centering // 2
            current_rotation = predicted_rotation # Keep original predicted rotation
            print(f"尝试 {attempt}: 使用居中 x={current_x}, 旋转={current_rotation}")
        elif attempt == 2: # Try original predicted_x with next rotation from original predicted_rotation
            current_x = predicted_x # Keep original predicted x
            current_rotation = (predicted_rotation + 1) % 4
            print(f"尝试 {attempt}: 使用原始 x={current_x}, 新旋转={current_rotation}")
        elif attempt == 3: # Exhaustive search
            print(f"尝试 {attempt}: _compute_valid_move 穷举搜索任何有效移动...")
            for r_idx in range(4): # Iterate all 4 rotations
                rotated_piece_exhaustive = deepcopy(piece)
                for _ in range(r_idx):
                    rotated_piece_exhaustive = rotate(rotated_piece_exhaustive)
                
                width_exhaustive = len(rotated_piece_exhaustive[0]) if rotated_piece_exhaustive and len(rotated_piece_exhaustive) > 0 and len(rotated_piece_exhaustive[0]) > 0 else 1
                
                # Iterate x from far left to far right
                for x_ex in range(-width_exhaustive + 1, len(board[0])):
                    if check(board, rotated_piece_exhaustive, [x_ex, 0]): # Check if valid at top
                        y_ex_final = 0
                        while y_ex_final < len(board) - 1 and \
                              check(board, rotated_piece_exhaustive, [x_ex, y_ex_final + 1]):
                            y_ex_final += 1
                        
                        if check(board, rotated_piece_exhaustive, [x_ex, y_ex_final]): # Final check
                            print(f"穷举搜索成功: x={x_ex}, y={y_ex_final}, 旋转={r_idx}")
                            return {'x': x_ex, 'y': y_ex_final, 'rotation': r_idx}
            
            print("警告: _compute_valid_move 穷举搜索未能找到任何有效移动。将进入下一尝试（可能触发最大尝试次数）。")
            return self._compute_valid_move(board, piece, predicted_x, predicted_rotation, attempt + 1)

        # Common logic for attempts 0, 1, 2 (after current_x, current_rotation are set for the attempt)
        # For attempt 0, current_x = predicted_x, current_rotation = predicted_rotation
        if attempt == 0:
            print(f"尝试 {attempt}: 使用预测 x={current_x}, 旋转={current_rotation}")

        final_rotated_piece = deepcopy(piece)
        for _ in range(current_rotation):
            final_rotated_piece = rotate(final_rotated_piece)

        if not check(board, final_rotated_piece, [current_x, 0]):
            print(f"尝试 {attempt} 失败: 初始位置 (x={current_x}, 旋转={current_rotation}) 无效。进入下一尝试。")
            return self._compute_valid_move(board, piece, predicted_x, predicted_rotation, attempt + 1)

        y_final_val = 0
        while y_final_val < len(board) - 1 and check(board, final_rotated_piece, [current_x, y_final_val + 1]):
            y_final_val += 1
        
        if check(board, final_rotated_piece, [current_x, y_final_val]):
            print(f"尝试 {attempt} 成功: x={current_x}, y={y_final_val}, 旋转={current_rotation}")
            return {'x': current_x, 'y': y_final_val, 'rotation': current_rotation}
        else:
            print(f"尝试 {attempt} 失败: 计算的最终位置 (x={current_x}, y={y_final_val}, 旋转={current_rotation}) 无效（尽管初始位置有效）。进入下一尝试。")
            return self._compute_valid_move(board, piece, predicted_x, predicted_rotation, attempt + 1)
    
    def create_state_vector(self, board, piece):
        """创建游戏状态向量，与训练时相同格式"""
        # 展平游戏板 (20x10 = 200)
        board_vector = np.array(board, dtype=np.float32).flatten()

        # 计算额外的启发式特征
        height_val = np.array([get_height(board)], dtype=np.float32)
        holes_val = np.array([count_holes(board)], dtype=np.float32)
        bumpiness_val = np.array([get_bumpiness(board)], dtype=np.float32)
        
        # 将piece转换为4x4矩阵
        piece_matrix = np.zeros((4, 4), dtype=np.float32)
        # Ensure piece is a numpy array for shape attribute
        if not isinstance(piece, np.ndarray):
            piece_array = np.array(piece)
        else:
            piece_array = piece
            
        h, w = piece_array.shape
        # 将piece放在4x4矩阵的中心
        start_h = (4 - h) // 2
        start_w = (4 - w) // 2
        piece_matrix[start_h:start_h+h, start_w:start_w+w] = piece_array
        piece_vector = piece_matrix.flatten()  # 4x4 = 16
        
        # 合并状态 (200 + 1 + 1 + 1 + 16 = 219)
        return np.concatenate([board_vector, height_val, holes_val, bumpiness_val, piece_vector])

def run_single_game_for_ai(ai_agent, initial_board_state=None, max_steps=500, game_id=0, debug_prints=False):
    """
    运行给定AI代理的单个游戏。
    Args:
        ai_agent (TetrisAI): 玩游戏的AI代理。
        initial_board_state (list, optional): 初始游戏板。如果为None，则创建空板。
        max_steps (int): 游戏的最大移动次数。
        game_id (int): 游戏ID，用于日志记录。
        debug_prints (bool): 是否打印详细的调试信息。
    Returns:
        tuple: (score, lines_cleared, moves_count, final_board_state, game_over_reason)
    """
    if initial_board_state is None:
        board = [[0 for _ in range(10)] for _ in range(20)]
    else:
        board = deepcopy(initial_board_state)
    
    score = 0
    lines_cleared_total = 0
    moves_count = 0
    game_over_reason = "Max steps reached"

    if debug_prints: print(f"--- 开始游戏 {game_id} ---")

    for move_idx in range(max_steps):
        current_piece_shape_config = random.choice(shapes) 
        current_piece = deepcopy(current_piece_shape_config)

        if debug_prints:
            print(f"\n游戏 {game_id}, 步数 {move_idx + 1}/{max_steps}")
            # print_board(board) # Assuming print_board is available and defined
            # print(f"当前方块: {current_piece}")

        # 检查新方块是否可以放置在标准起始位置
        start_x_nominal = len(board[0]) // 2 - (len(current_piece[0]) // 2 if current_piece and current_piece[0] else 0)
        if not check(board, current_piece, [start_x_nominal, 0]):
            if debug_prints: print(f"游戏结束: 新方块无法放置在起始位置。游戏 {game_id}, 步数 {move_idx + 1}")
            game_over_reason = "Cannot place new piece"
            break 
        
        predicted_move = ai_agent.predict_move(board, current_piece)

        # predicted_move should always contain a dictionary due to _compute_valid_move's fallbacks
        if debug_prints:
            print(f"AI预测移动: x={predicted_move['x']}, y={predicted_move['y']} (calculated), rotation={predicted_move['rotation']}")

        final_x = predicted_move['x']
        final_y = predicted_move['y'] 
        final_rotation = predicted_move['rotation']

        rotated_piece = deepcopy(current_piece)
        for _ in range(final_rotation):
            rotated_piece = rotate(rotated_piece)

        # 关键检查: AI选择的最终移动是否真的有效? _compute_valid_move应该保证这一点。
        if not check(board, rotated_piece, [final_x, final_y]):
            if debug_prints:
                print(f"错误: AI选择的最终移动无效! x={final_x}, y={final_y}, rot={final_rotation}. 游戏 {game_id}, 步数 {move_idx + 1}")
                # print_board(board)
                # print("Piece to place:", rotated_piece)
            game_over_reason = "AI produced an invalid final move after _compute_valid_move"
            break 
        
        join_matrix(board, rotated_piece, [final_x, final_y])
        
        lines_cleared_this_step = 0
        board, lines_cleared_this_step = clear_rows(board) 
        # print(f"消除行数: {lines_cleared_this_step}") # Debug print

        if lines_cleared_this_step == 1: score += 40
        elif lines_cleared_this_step == 2: score += 100
        elif lines_cleared_this_step == 3: score += 300
        elif lines_cleared_this_step >= 4: score += 1200
        score += 1 # Score for surviving a step

        lines_cleared_total += lines_cleared_this_step
        moves_count += 1

        if debug_prints:
            print(f"移动执行完毕。消除行数: {lines_cleared_this_step}, 总分数: {score}, 总消除行: {lines_cleared_total}")
            # print_board(board)

    if debug_prints:
        print(f"--- 游戏 {game_id} 结束 ---")
        print(f"最终得分: {score}, 总消除行: {lines_cleared_total}, 总步数: {moves_count}")
        print(f"结束原因: {game_over_reason}")
        # print_board(board)
        
    return score, lines_cleared_total, moves_count, board, game_over_reason

def main():
    # Define base directories
    # supervised_learning directory
    supervised_learning_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # core directory (current file's directory)
    core_dir = os.path.dirname(os.path.abspath(__file__))

    models_base_dir = os.path.join(supervised_learning_dir, "models")
    training_data_base_dir = os.path.join(supervised_learning_dir, "training_data")
    training_logs_base_dir = os.path.join(core_dir, "training_logs") # Logs specific to core operations

    # Ensure directories exist
    for dir_path in [models_base_dir, training_data_base_dir, training_logs_base_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")

    try:
        print("=" * 50)
        print("俄罗斯方块监督学习系统 v2.0")
        print("=" * 50)
        
        print("请选择模式:")
        print("1. 收集训练数据并训练模型")
        print("2. 加载已有数据并训练模型")
        print("3. 测试训练好的模型")
        print("4. 数据分析与可视化 (训练历史)")
        print("5. 使用不同模型对比")
        print("0. 退出")
        mode = input("请输入选项 (0-5,默认为1): ") or "1"
        
        if mode == "1":
            print("\\n=== 数据收集与模型训练 ===")
            print("请选择数据收集量:")
            print("1. 快速测试 (5局游戏,每局50步, 20轮训练)")
            print("2. 标准训练 (20局游戏,每局100步, 100轮训练)")
            print("3. 大规模训练 (50局游戏,每局200步, 150轮训练)")
            data_option = input("请选择数据量 (1/2/3,默认为2): ") or "2"
            
            collector_params = {}
            train_epochs = 100

            if data_option == "1":
                collector_params = {"num_games": 5, "max_moves": 50, "timeout": 30}
                train_epochs = 20
            elif data_option == "3":
                collector_params = {"num_games": 50, "max_moves": 200, "timeout": 180}
                train_epochs = 150
            else: # Default to option 2
                collector_params = {"num_games": 20, "max_moves": 100, "timeout": 120}
                train_epochs = 100
            
            collector = TetrisDataCollector(**collector_params)
            game_states, moves = collector.collect_data()
            
            if game_states is not None and len(game_states) > 0 and \
               moves is not None and len(moves) > 0:
                print(f"成功收集 {len(game_states)} 个数据点。")
                train_network(game_states, moves, num_epochs=train_epochs)
            else:
                print("未能收集到足够的训练数据或数据收集被中断。")

        elif mode == "2":
            print("\\n=== 加载数据并训练模型 ===")
            data_filename = input(f"请输入训练数据文件名 (默认为 tetris_training_data.npz, 位于 {training_data_base_dir}): ") or 'tetris_training_data.npz'
            
            game_states_data, moves_data = TetrisDataset.load_from_file(filename=data_filename) # Assumes load_from_file handles path correctly
            
            # Check if data loading was successful and data is valid
            if game_states_data is not None and moves_data is not None and \
               len(game_states_data) > 0 and len(moves_data) > 0 and \
               len(game_states_data) == len(moves_data):
                
                print(f"成功加载数据: {len(game_states_data)} 个状态, {len(moves_data)} 个移动")
                train_network(game_states_data, moves_data, num_epochs=100, batch_size=128, patience=20) 
            else:
                loaded_gs_len = len(game_states_data) if game_states_data is not None else 0
                loaded_m_len = len(moves_data) if moves_data is not None else 0
                print(f"没有加载到有效数据、数据为空或状态与移动数量不匹配 (状态: {loaded_gs_len}, 移动: {loaded_m_len})，无法训练。请先收集数据 (模式1)。")

        elif mode == "3":
            print("\\n=== 测试模型 ===")
            default_model_name = "tetris_model_best.pth"
            model_name = input(f"请输入模型文件名 (默认为 {default_model_name}, 位于 {models_base_dir}): ") or default_model_name
            model_path = os.path.join(models_base_dir, model_name)

            if not os.path.exists(model_path):
                print(f"错误: 模型文件未找到 {model_path}")
            else:
                try:
                    ai = TetrisAI(model_path=model_path)
                    initial_board = [[0 for _ in range(10)] for _ in range(20)]
                    num_test_games = int(input("请输入测试游戏局数 (默认为5): ") or "5")
                    
                    all_scores, all_lines, all_moves = [], [], []
                    for i in range(num_test_games):
                        print(f"--- 开始游戏 {i+1}/{num_test_games} ---")
                        current_board = deepcopy(initial_board)
                        score, lines, num_moves, final_board, game_over_reason = run_single_game_for_ai(ai, current_board, max_steps=500)
                        print(f"游戏 {i+1} 完成: 得分={score}, 行数={lines}, 步数={num_moves}")
                        if num_test_games == 1 or input("显示最终棋盘吗? (y/N): ").lower() == 'y':
                             print("最终棋盘状态:")
                             print_board(final_board)
                        all_scores.append(score)
                        all_lines.append(lines)
                        all_moves.append(num_moves)
                    
                    print("\\n--- 测试总结 ---")
                    if all_scores: # Avoid error if no games were run (e.g. num_test_games = 0)
                        print(f"平均得分: {np.mean(all_scores):.2f}")
                        print(f"平均行数: {np.mean(all_lines):.2f}")
                        print(f"平均步数: {np.mean(all_moves):.2f}")
                    else:
                        print("没有进行任何测试游戏。")

                except Exception as e_test:
                    print(f"测试模型时出错: {e_test}")
                    traceback.print_exc()

        elif mode == "4":
            print("\\n=== 数据分析与可视化 (训练历史) ===")
            history_filename = input("请输入训练历史文件名 (默认为 training_history.json): ") or "training_history.json"
            history_file_path = os.path.join(training_logs_base_dir, history_filename)

            if not os.path.exists(history_file_path):
                print(f"错误: 找不到训练历史文件 {history_file_path}")
            else:
                try:
                    with open(history_file_path, 'r') as f:
                        history = json.load(f)
                    
                    train_loss = history.get("train_loss", [])
                    val_loss = history.get("val_loss", [])
                    lr_history = history.get("learning_rates", [])
                    
                    if not train_loss:
                        print("错误: 训练历史数据中未找到训练损失。")
                        return

                    epochs = range(1, len(train_loss) + 1)

                    plt.style.use('seaborn-v0_8-whitegrid')
                    fig, ax1 = plt.subplots(figsize=(12, 7))

                    color = 'tab:red'
                    ax1.set_xlabel('轮数 (Epochs)')
                    ax1.set_ylabel('损失 (Loss)', color=color)
                    ax1.plot(epochs, train_loss, label='训练损失 (Train Loss)', color='tab:blue', linestyle='--')
                    if val_loss: # Only plot if val_loss exists
                        ax1.plot(epochs, val_loss, label='验证损失 (Validation Loss)', color=color)
                    ax1.tick_params(axis='y', labelcolor=color)
                    ax1.legend(loc='upper left')
                    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

                    if lr_history:
                        ax2 = ax1.twinx()
                        color = 'tab:green'
                        ax2.set_ylabel('学习率 (Learning Rate)', color=color)
                        # Ensure lr_history has same length as epochs for plotting
                        ax2.plot(epochs[:len(lr_history)], lr_history, label='学习率 (Learning Rate)', color=color, linestyle=':')
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.legend(loc='upper right')
                    
                    plt.title('训练与验证损失以及学习率变化', fontsize=16)
                    fig.tight_layout()
                    
                    plot_save_path = os.path.join(training_logs_base_dir, "training_performance.png")
                    plt.savefig(plot_save_path)
                    print(f"图表已保存到: {plot_save_path}")
                    if input("是否现在显示图表? (Y/n): ").lower() != 'n':
                        plt.show()

                except json.JSONDecodeError:
                    print(f"错误: 训练历史文件 {history_file_path} 格式不正确。")
                except Exception as e_hist:
                    print(f"分析数据时出错: {e_hist}")
                    traceback.print_exc()
        
        elif mode == "5":
            print("\\n=== 模型对比 ===")
            default_model1_name = "tetris_model.pth"
            default_model2_name = "tetris_model_best.pth"

            model1_name = input(f"请输入模型1文件名 (默认为 {default_model1_name}, 位于 {models_base_dir}): ") or default_model1_name
            model1_path = os.path.join(models_base_dir, model1_name)
            
            model2_name = input(f"请输入模型2文件名 (默认为 {default_model2_name}, 位于 {models_base_dir}): ") or default_model2_name
            model2_path = os.path.join(models_base_dir, model2_name)

            if not os.path.exists(model1_path):
                print(f"错误: 模型1文件未找到 {model1_path}")
                return # Exits main if file not found
            if not os.path.exists(model2_path):
                print(f"错误: 模型2文件未找到 {model2_path}")
                return # Exits main if file not found

            try:
                ai1 = TetrisAI(model_path=model1_path)
                ai2 = TetrisAI(model_path=model2_path)
                
                num_compare_games = int(input("请输入对比游戏局数 (默认为10): ") or "10")
                if num_compare_games <= 0:
                    print("对比游戏局数必须大于0。")
                    return

                max_steps_per_game = 500 

                results = {} # Initialize results dictionary
                initial_board_template = [[0 for _ in range(10)] for _ in range(20)]

                for model_name_iter, ai_agent, model_p in [(model1_name, ai1, model1_path), (model2_name, ai2, model2_path)]:
                    print(f"\\n--- 测试模型: {model_name_iter} (来自 {model_p}) ---")
                    current_model_scores, current_model_lines, current_model_moves = [], [], []
                    for i in range(num_compare_games):
                        print(f"  开始游戏 {i+1}/{num_compare_games} for {model_name_iter}")
                        current_board = deepcopy(initial_board_template) 
                        score, lines, num_moves, _, game_over_reason = run_single_game_for_ai(ai_agent, current_board, max_steps=max_steps_per_game)
                        print(f"  游戏 {i+1} ({model_name_iter}): 得分={score}, 行数={lines}, 步数={num_moves}")
                        current_model_scores.append(score)
                        current_model_lines.append(lines)
                        current_model_moves.append(num_moves)
                    
                    results[model_name_iter] = {
                        "scores": current_model_scores,
                        "lines": current_model_lines,
                        "moves": current_model_moves,
                        "avg_score": np.mean(current_model_scores) if current_model_scores else 0,
                        "avg_lines": np.mean(current_model_lines) if current_model_lines else 0,
                        "avg_moves": np.mean(current_model_moves) if current_model_moves else 0
                    }
                
                print("\\n--- 模型对比总结 ---")
                for model_name_res, res_data in results.items():
                    print(f"模型: {model_name_res}")
                    print(f"  平均得分: {res_data['avg_score']:.2f}")
                    print(f"  平均行数: {res_data['avg_lines']:.2f}")
                    print(f"  平均步数: {res_data['avg_moves']:.2f}")
                    print("-" * 30)

            except Exception as e_compare:
                print(f"对比模型时出错: {e_compare}")
                traceback.print_exc()

        elif mode == "0":
            print("退出程序。")
            
        else:
            print("无效选项，请输入0-5之间的数字。")
            
    except KeyboardInterrupt:
        print("\\n操作被用户中断。")
    except Exception as e:
        print(f"发生意外错误: {e}")
        traceback.print_exc()
    finally:
        print("\\n程序结束。")

if __name__ == '__main__':
    # 确保脚本可以从任何位置运行，并正确处理相对路径
    # 获取脚本所在的目录
    # current_script_path = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(current_script_path) # 假设 core 在 supervised_learning 下
    # sys.path.append(project_root) # 将项目根目录添加到Python路径，以便导入Tetris
    # print(f"Current working directory: {os.getcwd()}")
    # print(f"Sys.path: {sys.path}")
    main()
