#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tetris Supervised Learning System - Training Pipeline
简化版训练管道，专注于核心功能
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# 导入必要的组件
try:
    from tetris_supervised_fixed import TetrisDataset, TetrisNet
except ImportError as e:
    logging.error(f"Error importing required modules: {e}")
    print("请确保所有必要的脚本文件都在同一目录中")
    sys.exit(1)

class TrainingPipeline:
    """简化版Tetris AI训练管道"""
    
    def __init__(self, output_dir="training_output"):
        """初始化训练管道"""
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
        print(f"创建输出目录: {output_dir}")
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志记录"""
        log_file = os.path.join(self.output_dir, f'training_{self.timestamp}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def save_checkpoint(self, model, optimizer, epoch, val_loss, filepath):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, filepath)
        self.logger.info(f"保存检查点到 {filepath}")
        
    def train_model(self, train_data, train_targets, val_data=None, val_targets=None, 
                   epochs=50, batch_size=64, learning_rate=0.001):
        """训练模型的核心功能"""
        # 创建数据集
        train_dataset = TetrisDataset(train_data, train_targets)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        if val_data is not None and val_targets is not None:
            val_dataset = TetrisDataset(val_data, val_targets)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size
            )
        else:
            val_loader = None
            
        # 初始化模型和优化器
        model = TetrisNet()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f'轮次 {epoch+1}/{epochs} [批次 {batch_idx}/{len(train_loader)}] '
                        f'损失: {loss.item():.6f}'
                    )
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            if val_loader:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        val_loss += criterion(output, target).item()
                
                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_checkpoint(
                        model, optimizer, epoch, avg_val_loss,
                        os.path.join(self.output_dir, 'best_model.pth')
                    )
                
                self.logger.info(
                    f'轮次: {epoch+1}/{epochs}\n'
                    f'平均训练损失: {avg_train_loss:.6f}\n'
                    f'验证损失: {avg_val_loss:.6f}'
                )
            else:
                self.logger.info(
                    f'轮次: {epoch+1}/{epochs}\n'
                    f'平均训练损失: {avg_train_loss:.6f}'
                )
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    model, optimizer, epoch, avg_val_loss if val_loader else avg_train_loss,
                    os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
                )
        
        return model

    def run_training(self, options=None):
        """运行训练流程"""
        # 默认选项
        default_options = {
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "val_split": 0.2
        }
        
        # 更新选项
        if options:
            default_options.update(options)
        opts = default_options
        
        self.logger.info("开始训练流程...")
        self.logger.info("配置选项:")
        for k, v in opts.items():
            self.logger.info(f"{k}: {v}")
        
        try:
            # 加载数据
            self.logger.info("加载训练数据...")
            game_states, moves = TetrisDataset.load_from_file()
            
            if game_states is None or len(game_states) == 0:
                self.logger.error("没有找到训练数据")
                return False
            
            self.logger.info(f"加载了 {len(game_states)} 个训练样本")
            
            # 划分训练集和验证集
            split_idx = int(len(game_states) * (1 - opts["val_split"]))
            train_data = game_states[:split_idx]
            train_targets = moves[:split_idx]
            val_data = game_states[split_idx:]
            val_targets = moves[split_idx:]
            
            # 训练模型
            self.logger.info("开始训练模型...")
            model = self.train_model(
                train_data, train_targets,
                val_data, val_targets,
                epochs=opts["epochs"],
                batch_size=opts["batch_size"],
                learning_rate=opts["learning_rate"]
            )
            
            self.logger.info("训练完成！")
            return True
            
        except KeyboardInterrupt:
            self.logger.warning("训练被用户中断")
            return False
        except Exception as e:
            self.logger.error(f"训练过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    pipeline = TrainingPipeline()
    pipeline.run_training()

if __name__ == "__main__":
    main()
