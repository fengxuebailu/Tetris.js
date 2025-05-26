#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 增强版训练流程
用于创建更高质量的训练数据和模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import json
from Tetris import shapes, rotate, check, join_matrix, clear_rows, get_height, count_holes, get_bumpiness
from tetris_supervised_fixed import TetrisDataset, TetrisNet, train_network, save_model

class EnhancedDataCollector:
    """增强版数据收集器，使用策略混合和数据增强"""
    
    def __init__(self, num_games=100, max_moves=200, timeout=60):
        self.num_games = num_games
        self.max_moves = max_moves
        self.timeout = timeout
        self.game_states = []
        self.optimal_moves = []
        
        # 策略权重配置
        self.strategy_weights = {
            'evolved': {'weight': 0.7, 'enabled': True},  # 进化算法权重
            'height_based': {'weight': 0.2, 'enabled': True},  # 基于高度的策略
            'clearing': {'weight': 0.1, 'enabled': True}  # 专注于消行的策略
        }
    
    def load_weights(self, strategy='evolved'):
        """加载或创建不同策略的权重"""
        if strategy == 'evolved':
            try:
                with open('best_weights_evolved.json', 'r') as f:
                    weights = json.load(f)
                    # 确保所有值都是浮点数
                    return {k: float(v) for k, v in weights.items()}
            except:
                return {
                    'cleared_lines': 160.0,
                    'holes': -50.0,
                    'bumpiness': -20.0,
                    'height': -30.0
                }
        elif strategy == 'height_based':
            # 偏好较低的高度
            return {
                'cleared_lines': 100.0,
                'holes': -30.0,
                'bumpiness': -15.0,
                'height': -60.0  # 更重视高度
            }
        elif strategy == 'clearing':
            # 偏好消行
            return {
                'cleared_lines': 250.0,  # 更重视消行
                'holes': -40.0,
                'bumpiness': -10.0,
                'height': -20.0
            }
    
    def collect_enhanced_data(self):
        """使用多策略收集更全面的游戏数据"""
        print("开始增强版数据收集...")
        try:
            # 计算实际使用的策略
            enabled_strategies = [s for s, config in self.strategy_weights.items() 
                                if config['enabled']]
            
            if not enabled_strategies:
                print("错误: 没有启用任何策略")
                return None, None
                
            print(f"使用 {len(enabled_strategies)} 个收集策略:")
            for s in enabled_strategies:
                print(f"  - {s} (权重: {self.strategy_weights[s]['weight']})")
            
            # 分配游戏数
            strategy_games = {}
            remaining_games = self.num_games
            
            # 按权重分配游戏数
            total_weight = sum(self.strategy_weights[s]['weight'] 
                             for s in enabled_strategies)
            
            for strategy in enabled_strategies[:-1]:  # 除了最后一个
                games = int(self.num_games * (self.strategy_weights[strategy]['weight'] / total_weight))
                strategy_games[strategy] = games
                remaining_games -= games
            
            # 最后一个策略获取剩余的游戏数
            strategy_games[enabled_strategies[-1]] = remaining_games
            
            print("\n游戏分配:")
            for strategy, games in strategy_games.items():
                print(f"  {strategy}: {games}局")
            
            # 收集开始
            total_moves = 0
            start_time = time.time()
            
            # 对每个策略收集数据
            for strategy, num_games in strategy_games.items():
                print(f"\n使用 {strategy} 策略收集 {num_games} 局游戏数据...")
                weights = self.load_weights(strategy)
                print(f"权重: {weights}")
                
                for game in range(num_games):
                    print(f"收集游戏 {game+1}/{num_games} (策略: {strategy})")
                    board = [[0 for _ in range(10)] for _ in range(20)]
                    moves_count = 0
                    game_start_time = time.time()
                    
                    # 限制每个游戏的时间
                    while moves_count < self.max_moves:
                        if time.time() - game_start_time > self.timeout:
                            print(f"游戏 {game+1} 超时，已收集 {moves_count} 步")
                            break
                        
                        # 随机选择一个方块
                        current_piece = random.choice(shapes)
                        
                        # 使用策略查找最佳移动
                        move_data = self.find_best_move_optimized(board, current_piece, weights)
                        
                        if move_data is None:
                            print(f"游戏 {game+1} 无法继续移动")
                            break
                        
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
                            rotated_piece = rotate(rotated_piece)
                        
                        # 更新游戏板
                        join_matrix(board, rotated_piece, [move_data['x'], move_data['y']])
                        board, _ = clear_rows(board)
                        moves_count += 1
                        total_moves += 1
                        
                        # 每20步打印进度
                        if moves_count % 20 == 0:
                            print(f"  游戏 {game+1}: 已完成 {moves_count} 步")
                            
                    print(f"游戏 {game+1} 完成，收集了 {moves_count} 步，耗时: {time.time() - game_start_time:.1f}秒")
                    
                    # 检查总体超时
                    if time.time() - start_time > self.timeout * 2:
                        print("数据收集总体时间过长，提前结束")
                        break
            
            # 数据增强 (随机旋转和翻转)
            if len(self.game_states) > 0:
                print("\n进行数据增强...")
                original_count = len(self.game_states)
                self.apply_data_augmentation()
                print(f"数据增强后: {len(self.game_states)} 个样本 (增加了 {len(self.game_states) - original_count} 个)")

            # 转换为numpy数组并设定数据类型
            print(f"完成数据收集，共收集 {total_moves} 步原始数据，{len(self.game_states)} 个增强后的状态")
            print(f"总耗时: {time.time() - start_time:.1f}秒")
            
            game_states_array = np.array(self.game_states, dtype=np.float32)
            optimal_moves_array = np.array(self.optimal_moves, dtype=np.float32)
            
            # 保存训练数据
            TetrisDataset.save_to_file(game_states_array, optimal_moves_array)
            
            return game_states_array, optimal_moves_array
            
        except Exception as e:
            print(f"收集数据时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 即使发生错误，也尝试返回已收集的数据
            if len(self.game_states) > 0:
                return np.array(self.game_states, dtype=np.float32), np.array(self.optimal_moves, dtype=np.float32)
            raise
    
    def apply_data_augmentation(self):
        """应用数据增强技术"""
        augmented_states = []
        augmented_moves = []
        
        # 随机选择一部分样本进行增强
        augment_count = min(len(self.game_states) // 3, 100)  # 最多增强1/3的数据，但不超过100个
        indices = random.sample(range(len(self.game_states)), augment_count)
        
        for idx in indices:
            state = self.game_states[idx]
            move = self.optimal_moves[idx]
            
            # 1. 水平翻转游戏板 (只对游戏板部分操作，不包括方块信息)
            flipped_state = state.copy()
            board_part = flipped_state[:200].reshape(20, 10)
            flipped_board = np.fliplr(board_part).flatten()
            flipped_state[:200] = flipped_board
            
            # 相应地调整移动
            flipped_move = move.copy()
            flipped_move[0] = 9 - move[0]  # 翻转x坐标
            
            augmented_states.append(flipped_state)
            augmented_moves.append(flipped_move)
            
            # 2. 轻微随机噪声 (只对游戏板添加少量随机噪声，可能会让空白格子看起来不那么空)
            noisy_state = state.copy()
            noise = np.random.normal(0, 0.05, 200)  # 小幅度的高斯噪声
            board_part = noisy_state[:200]
            
            # 只给空白区域 (0值) 添加噪声
            mask = (board_part == 0)
            board_part[mask] += noise[mask]
            board_part = np.clip(board_part, 0, 1)  # 确保值在0-1范围内
            noisy_state[:200] = board_part
            
            augmented_states.append(noisy_state)
            augmented_moves.append(move.copy())  # 移动不变
            
        # 将增强的数据添加到原始数据中
        self.game_states.extend(augmented_states)
        self.optimal_moves.extend(augmented_moves)
    
    def create_state_vector(self, board, piece):
        """创建游戏状态向量"""
        # 展平游戏板 (20x10 = 200)
        board_vector = np.array(board, dtype=np.float32).flatten()
        
        # 将piece转换为4x4矩阵
        piece_matrix = np.zeros((4, 4), dtype=np.float32)
        piece = np.array(piece)
        h, w = piece.shape
        start_h = (4 - h) // 2
        start_w = (4 - w) // 2
        piece_matrix[start_h:start_h+h, start_w:start_w+w] = piece
        piece_vector = piece_matrix.flatten()  # 4x4 = 16
        
        # 合并状态 (200 + 16 = 216)
        return np.concatenate([board_vector, piece_vector])
    
    def find_best_move_optimized(self, board, piece, weights):
        """寻找最佳移动的优化版本"""
        best_score = float('-inf')
        best_move = None
        start_time = time.time()
        
        # 优化：根据方块形状决定需要测试的旋转次数
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
        
        # 预计算所有可能的旋转
        rotated_pieces = []
        current = deepcopy(piece)
        for r in range(max_rotations):
            rotated_pieces.append(current)
            current = rotate(current)
            
        # 评估所有位置
        for rotation, current_piece in enumerate(rotated_pieces):
            width = len(current_piece[0])
            height = len(current_piece)
            
            # 设置合理的x范围
            min_x = max(-2, -width+1)
            max_x = min(len(board[0]), len(board[0])+2)
            
            for x in range(min_x, max_x):
                # 检查是否超时
                if time.time() - start_time > 0.5:  # 限制0.5秒
                    if best_move is None:
                        # 如果还没找到最佳移动，返回一个简单的默认移动
                        for test_x in range(len(board[0]) - len(current_piece[0]) + 1):
                            y = 0
                            while y < len(board) and check(board, current_piece, [test_x, y+1]):
                                y += 1
                            if check(board, current_piece, [test_x, y]):
                                return {'x': test_x, 'y': y, 'rotation': rotation}
                    return best_move
                
                # 如果初始位置不合法，跳过
                if not check(board, current_piece, [x, 0]):
                    continue
                
                # 快速下落
                y = 0
                while y < len(board) and check(board, current_piece, [x, y+1]):
                    y += 1
                
                # 创建临时板并放置方块
                temp_board = [row[:] for row in board]
                join_matrix(temp_board, current_piece, [x, y])
                new_board, cleared = clear_rows(temp_board)
                
                # 计算评分
                score = (weights['cleared_lines'] * cleared +
                        weights['holes'] * count_holes(new_board) +
                        weights['bumpiness'] * get_bumpiness(new_board) +
                        weights['height'] * get_height(new_board))
                
                if score > best_score:
                    best_score = score
                    best_move = {
                        'x': x,
                        'y': y,
                        'rotation': rotation
                    }
        
        return best_move

class ImprovedTetrisNet(nn.Module):
    """改进版俄罗斯方块AI神经网络，增加了残差连接和批标准化"""
    
    def __init__(self):
        super(ImprovedTetrisNet, self).__init__()
        # 输入特征: 游戏板状态(200)和当前方块(16)
        
        # 游戏板特征提取
        self.board_features = nn.Sequential(
            nn.Linear(200, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(192, 144)
        )
        
        # 方块特征提取
        self.piece_features = nn.Sequential(
            nn.Linear(16, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1)
        )
        
        # 合并后的处理网络（包含残差连接）
        self.fc1 = nn.Linear(168, 128)  # 144 + 24 = 168
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 2)  # 输出: x位置, 旋转角度
        
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 分离游戏板和当前方块的特征
        board_input = x[:, :200]  # 前200个特征是游戏板
        piece_input = x[:, 200:]  # 后16个特征是方块
        
        # 各自通过特征提取网络
        board_feats = self.board_features(board_input)
        piece_feats = self.piece_features(piece_input)
        
        # 合并特征
        combined = torch.cat([board_feats, piece_feats], dim=1)
        
        # 前向传播，残差连接
        x1 = self.leaky_relu(self.bn1(self.fc1(combined)))
        x2 = self.leaky_relu(self.bn2(self.fc2(x1)))
        x2 = x2 + x1  # 残差连接
        x2 = self.dropout(x2)
        
        x3 = self.leaky_relu(self.bn3(self.fc3(x2)))
        x4 = self.leaky_relu(self.bn4(self.fc4(x3)))
        out = self.fc5(x4)
        
        return out

class EnhancedModelTrainer:
    """增强版模型训练器，包含更多的训练选项和可视化"""
    
    def __init__(self, output_dir="training_results", model_architecture="improved"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.model_architecture = model_architecture
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": []
        }
    
    def train_with_enhanced_options(self, game_states, moves, options=None):
        """使用增强选项训练模型"""
        # 默认选项
        default_options = {
            "num_epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "patience": 15,
            "weight_decay": 1e-5,
            "use_lr_scheduler": True,
            "train_split": 0.8,
            "model_prefix": "tetris_model"
        }
        
        # 更新选项
        if options:
            default_options.update(options)
        
        opts = default_options
        print(f"\n=== 使用增强训练选项训练模型 ===")
        print(f"模型架构: {self.model_architecture}")
        print(f"训练数据: {len(game_states)} 个样本")
        print(f"轮数: {opts['num_epochs']}")
        print(f"批次大小: {opts['batch_size']}")
        print(f"学习率: {opts['learning_rate']}")
        print(f"早停耐心值: {opts['patience']}")
        print(f"权重衰减: {opts['weight_decay']}")
        
        # 创建数据集
        dataset = TetrisDataset(game_states, moves)
        
        # 划分训练集和验证集
        train_size = int(opts['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=opts['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=opts['batch_size'])
        
        # 初始化模型
        if self.model_architecture == "improved":
            model = ImprovedTetrisNet()
            print("使用改进版神经网络架构")
        else:
            model = TetrisNet()
            print("使用标准神经网络架构")
            
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=opts['learning_rate'],
            weight_decay=opts['weight_decay']
        )
        
        # 学习率调度器
        if opts['use_lr_scheduler']:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-6
            )
        else:
            scheduler = None
        
        print("\n开始训练模型...")
        print(f"训练数据: {train_size} 个样本, 验证数据: {val_size} 个样本")
        print(f"批次大小: {opts['batch_size']}, 最大轮数: {opts['num_epochs']}")
        
        # 早停策略变量
        best_val_loss = float('inf')
        best_model = None
        epochs_no_improve = 0
        
        # 训练循环
        start_time = time.time()
        for epoch in range(opts['num_epochs']):
            epoch_start_time = time.time()
            
            # 训练阶段
            model.train()
            train_loss = 0
            for batch_states, batch_moves in train_loader:
                # 前向传播
                outputs = model(batch_states)
                loss = criterion(outputs, batch_moves)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_states, batch_moves in val_loader:
                    outputs = model(batch_states)
                    loss = criterion(outputs, batch_moves)
                    val_loss += loss.item()
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # 保存历史
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["learning_rates"].append(optimizer.param_groups[0]["lr"])
            
            # 更新学习率调度器
            if scheduler:
                scheduler.step(avg_val_loss)
            
            # 记录时间和输出信息
            epoch_time = time.time() - epoch_start_time
            print(f'轮数 {epoch+1}/{opts["num_epochs"]}, 训练损失: {avg_train_loss:.4f}, '
                 f'验证损失: {avg_val_loss:.4f}, 耗时: {epoch_time:.2f}秒')
            
            # 检查是否有改进
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = deepcopy(model.state_dict())
                epochs_no_improve = 0
                # 保存最佳模型
                save_model(model, f"{opts['model_prefix']}_best.pth")
                print(f"发现更好的模型，已保存!")
            else:
                epochs_no_improve += 1
            
            # 每10轮保存一次模型
            if (epoch + 1) % 10 == 0:
                save_model(model, f"{opts['model_prefix']}_epoch_{epoch+1}.pth")
                # 保存训练历史
                self.save_history()
                # 生成损失曲线
                self.plot_training_curves()
            
            # 早停检查
            if epochs_no_improve >= opts['patience']:
                print(f"早停: {opts['patience']}轮内验证损失没有改善")
                break
        
        # 恢复最佳模型
        if best_model is not None:
            model.load_state_dict(best_model)
        
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"训练完成! 总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        
        # 保存最终模型
        final_model_path = f"{opts['model_prefix']}_final_{self.timestamp}.pth"
        save_model(model, final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
        
        # 保存训练历史
        self.save_history()
        
        # 生成最终的训练曲线
        self.plot_training_curves()
        
        return model, best_val_loss
    
    def save_history(self):
        """保存训练历史数据"""
        history_path = f"{self.output_dir}/training_history_{self.timestamp}.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f)
        print(f"训练历史已保存到: {history_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        try:
            epochs = range(1, len(self.history["train_loss"]) + 1)
            
            plt.figure(figsize=(15, 5))
            
            # 训练和验证损失
            plt.subplot(1, 2, 1)
            plt.plot(epochs, self.history["train_loss"], "b-", label="训练损失")
            plt.plot(epochs, self.history["val_loss"], "r-", label="验证损失")
            plt.title("训练和验证损失")
            plt.xlabel("轮数")
            plt.ylabel("损失")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 学习率变化
            plt.subplot(1, 2, 2)
            plt.plot(epochs, self.history["learning_rates"], "g-")
            plt.title("学习率变化")
            plt.xlabel("轮数")
            plt.ylabel("学习率")
            plt.yscale("log")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/training_curves_{self.timestamp}.png")
            plt.close()
            
            print(f"训练曲线已保存到: {self.output_dir}/training_curves_{self.timestamp}.png")
        except Exception as e:
            print(f"绘制训练曲线时出错: {e}")

def main():
    """增强版训练流程的主函数"""
    try:
        print("=" * 50)
        print("俄罗斯方块监督学习系统 - 增强版训练流程")
        print("=" * 50)
        
        # 命令行参数
        import argparse
        parser = argparse.ArgumentParser(description="俄罗斯方块监督学习系统增强训练脚本")
        parser.add_argument('--collect', action='store_true', help='收集新的训练数据')
        parser.add_argument('--games', type=int, default=100, help='收集数据的游戏数量')
        parser.add_argument('--merge', action='store_true', help='合并新收集的数据与已有数据')
        parser.add_argument('--model', choices=['standard', 'improved'], default='improved', help='模型架构')
        parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
        parser.add_argument('--batch', type=int, default=64, help='批次大小')
        args = parser.parse_args()
        
        # 检查CUDA可用性
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 收集数据（如果需要）
        if args.collect:
            print("\n=== 收集训练数据 ===")
            collector = EnhancedDataCollector(num_games=args.games)
            game_states, moves = collector.collect_enhanced_data()
            
            # 如果需要合并数据
            if args.merge and game_states is not None:
                try:
                    prev_states, prev_moves = TetrisDataset.load_from_file()
                    if prev_states is not None and len(prev_states) > 0:
                        print(f"合并现有的 {len(prev_states)} 个样本和新收集的 {len(game_states)} 个样本")
                        game_states = np.concatenate([prev_states, game_states])
                        moves = np.concatenate([prev_moves, moves])
                        TetrisDataset.save_to_file(game_states, moves)
                        print(f"数据合并完成，总共有 {len(game_states)} 个样本")
                except Exception as e:
                    print(f"合并数据失败: {e}")
        else:
            # 加载已有数据
            game_states, moves = TetrisDataset.load_from_file()
            
        if game_states is not None and len(game_states) > 0:
            print(f"\n=== 使用 {len(game_states)} 个样本训练模型 ===")
            
            # 训练模型
            trainer = EnhancedModelTrainer(model_architecture=args.model)
            options = {
                "num_epochs": args.epochs,
                "batch_size": args.batch,
                "model_prefix": "tetris_model_enhanced" if args.model == "improved" else "tetris_model"
            }
            
            model, best_loss = trainer.train_with_enhanced_options(game_states, moves, options)
            print(f"模型训练完成! 最佳损失: {best_loss:.4f}")
            
            # 打印测试命令提示
            print("\n要测试新训练的模型，请运行:")
            print(f"python test_models.py {options['model_prefix']}_best.pth")
        else:
            print("错误: 没有可用的训练数据")
    
    except Exception as e:
        print(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    # 如果没有命令行参数，添加--help参数显示帮助信息
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    main()
