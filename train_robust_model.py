#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 健壮模型训练脚本
专门用于训练解决无效移动问题的模型
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
import sys
import argparse
from Tetris import shapes, rotate, check, join_matrix, clear_rows, get_height, count_holes, get_bumpiness

# 确保可以导入必要的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
try:
    from tetris_supervised_fixed import TetrisDataset, TetrisNet, save_model
    from enhanced_training import ImprovedTetrisNet, EnhancedDataCollector
except ImportError as e:
    print(f"导入模块出错: {e}")
    sys.exit(1)

class RobustModelTrainer:
    """健壮型模型训练器，专注于减少无效移动"""
    
    def __init__(self, output_dir="robust_models"):
        """初始化训练器"""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": []
        }
    
    def collect_enhanced_data(self, num_games=100, max_moves=200):
        """使用增强数据收集器收集训练数据"""
        print(f"\n=== 收集健壮型训练数据 (共 {num_games} 局) ===")
        collector = EnhancedDataCollector(num_games=num_games, max_moves=max_moves)
        game_states, moves = collector.collect_enhanced_data()
        
        # 应用数据增强
        collector.apply_data_augmentation()
        
        # 生成更多的边界情况数据
        print("\n=== 生成边界情况训练数据 ===")
        self._generate_edge_case_data(collector)
        
        return collector.game_states, collector.optimal_moves
    
    def _generate_edge_case_data(self, collector, num_samples=100):
        """生成特别关注边界情况的数据"""
        print(f"生成 {num_samples} 个边界情况数据样本")
        
        for i in range(num_samples):
            # 创建具有挑战性的游戏板
            board = [[0 for _ in range(10)] for _ in range(20)]
            
            # 1. 在游戏板底部创建不规则形状
            for col in range(10):
                height = random.randint(5, 10)
                for row in range(20-height, 20):
                    board[row][col] = 1
            
            # 2. 创建一些孔洞
            for _ in range(random.randint(3, 8)):
                col = random.randint(0, 9)
                row = random.randint(10, 18)
                if board[row][col] == 1:
                    board[row][col] = 0
            
            # 3. 创建一些边界情况 - 左右边界有高墙
            for row in range(10, 20):
                if random.random() < 0.7:  # 70%的概率
                    board[row][0] = 1  # 左侧墙
                if random.random() < 0.7:
                    board[row][9] = 1  # 右侧墙
            
            # 对每个形状计算最佳移动
            for piece in shapes:
                # 使用收集器的find_best_move获取专家移动
                state_vector, optimal_move = collector.find_best_move(board, piece)
                
                # 添加到数据集
                collector.game_states.append(state_vector)
                collector.optimal_moves.append(optimal_move)
        
        print(f"边界情况数据生成完成，总共 {len(collector.game_states)} 个样本")
    
    def train_robust_model(self, game_states, moves, options=None):
        """训练健壮型模型"""
        # 默认选项
        default_options = {
            "num_epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.0005,
            "patience": 20,
            "weight_decay": 2e-5,
            "use_lr_scheduler": True,
            "train_split": 0.8,
            "model_type": "improved",  # 使用改进型网络
            "model_prefix": "tetris_robust"
        }
        
        # 更新选项
        if options:
            default_options.update(options)
        
        opts = default_options
        print(f"\n=== 开始训练健壮型模型 ===")
        print(f"模型类型: {opts['model_type']}")
        print(f"训练数据: {len(game_states)} 个样本")
        print(f"批次大小: {opts['batch_size']}")
        print(f"初始学习率: {opts['learning_rate']}")
        
        # 创建数据集
        dataset = TetrisDataset(game_states, moves)
        
        # 划分训练集和验证集
        train_size = int(opts['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=opts['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=opts['batch_size'])
        
        # 初始化模型
        if opts['model_type'] == "improved":
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
                self._save_history(f"{opts['model_prefix']}_history_{self.timestamp}.json")
                # 生成损失曲线
                self._plot_training_curves(opts['model_prefix'])
            
            # 早停检查
            if epochs_no_improve >= opts['patience']:
                print(f"早停: {opts['patience']}轮内验证损失没有改善")
                break
        
        # 恢复最佳模型
        if best_model is not None:
            model.load_state_dict(best_model)
        
        # 保存最终模型
        final_model_path = f"{opts['model_prefix']}_final.pth"
        save_model(model, final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
        
        # 保存训练历史和图表
        self._save_history(f"{opts['model_prefix']}_history_final.json")
        self._plot_training_curves(opts['model_prefix'])
        
        return model, best_val_loss
    
    def _save_history(self, filename):
        """保存训练历史"""
        history_path = f"{self.output_dir}/{filename}"
        with open(history_path, "w") as f:
            json.dump(self.history, f)
        print(f"训练历史已保存到: {history_path}")
    
    def _plot_training_curves(self, prefix):
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
            curve_path = f"{self.output_dir}/{prefix}_curves_{self.timestamp}.png"
            plt.savefig(curve_path)
            plt.close()
            
            print(f"训练曲线已保存到: {curve_path}")
        except Exception as e:
            print(f"绘制训练曲线时出错: {e}")

def validate_model(model_path, num_tests=20):
    """验证模型的有效移动率"""
    print(f"\n=== 验证模型 {model_path} 的有效移动率 ===")
    
    # 导入诊断工具
    from diagnose_invalid_moves import diagnose_moves
    
    # 验证模型
    stats = diagnose_moves(model_path, num_tests)
    
    if stats:
        valid_rate = stats['valid'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"\n模型 {model_path} 的有效移动率: {valid_rate:.1f}%")
        print(f"有效移动: {stats['valid']}/{stats['total']}")
        
        return valid_rate, stats
    
    return None, None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="俄罗斯方块健壮模型训练器")
    parser.add_argument('--collect', action='store_true', help='收集新的训练数据')
    parser.add_argument('--games', type=int, default=100, help='收集数据的游戏数量')
    parser.add_argument('--merge', action='store_true', help='合并新收集的数据与已有数据')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=64, help='批次大小')
    parser.add_argument('--validate', action='store_true', help='训练后验证模型')
    args = parser.parse_args()
    
    try:
        print("=" * 50)
        print("俄罗斯方块健壮模型训练系统")
        print("=" * 50)
        
        trainer = RobustModelTrainer()
        
        # 收集或加载数据
        if args.collect:
            # 收集新数据
            game_states, moves = trainer.collect_enhanced_data(num_games=args.games)
            
            # 如果需要合并数据
            if args.merge:
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
            
            # 保存新数据
            TetrisDataset.save_to_file(game_states, moves)
        else:
            # 加载已有数据
            game_states, moves = TetrisDataset.load_from_file()
            
        if game_states is not None and len(game_states) > 0:
            # 训练模型
            print(f"\n=== 使用 {len(game_states)} 个样本训练健壮模型 ===")
            
            options = {
                "num_epochs": args.epochs,
                "batch_size": args.batch,
                "model_prefix": "tetris_robust"
            }
            
            model, best_loss = trainer.train_robust_model(game_states, moves, options)
            print(f"模型训练完成! 最佳损失: {best_loss:.4f}")
            
            # 如果需要验证模型
            if args.validate:
                valid_rate, stats = validate_model("tetris_robust_best.pth", 30)
                if valid_rate is not None:
                    if valid_rate < 95:
                        print("警告: 模型的有效移动率低于95%，建议重新训练或调整模型")
                    else:
                        print(f"模型表现良好，有效移动率为 {valid_rate:.1f}%")
            
            # 打印测试命令提示
            print("\n要测试新训练的模型，请运行:")
            print("python test_models.py tetris_robust_best.pth")
        else:
            print("错误: 没有可用的训练数据")
    
    except Exception as e:
        print(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
