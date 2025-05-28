#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 训练工具类
提供训练过程管理、数据处理和评估功能
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt

class TrainingManager:
    """训练过程管理器"""
    
    def __init__(self, output_dir: str = "training_output"):
        """初始化训练管理器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = self._setup_logger()
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.training_interrupted = False
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f'training_{int(time.time())}')
        logger.setLevel(logging.INFO)
          # File handler with UTF-8 encoding
        fh = logging.FileHandler(os.path.join(self.output_dir, 'training.log'), encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def prepare_data(self, 
                    dataset: torch.utils.data.Dataset,
                    batch_size: int,
                    val_split: float = 0.2
                    ) -> Tuple[DataLoader, DataLoader]:
        """准备训练和验证数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            val_split: 验证集比例
            
        Returns:
            train_loader, val_loader: 训练和验证数据加载器
        """
        dataset_size = len(dataset)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )
        
        self.logger.info(f"数据集大小: {dataset_size}")
        self.logger.info(f"训练集: {train_size} 样本")
        self.logger.info(f"验证集: {val_size} 样本")
        
        return train_loader, val_loader
        
    def train(self,
              model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int,
              learning_rate: float = 0.001,
              patience: int = 15
              ) -> Tuple[nn.Module, float]:
        """训练模型
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            patience: 早停耐心值
            
        Returns:
            model: 训练好的模型
            best_val_loss: 最佳验证损失
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        early_stopping_counter = 0
        best_model_state = None
        
        try:
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_loss = self._train_epoch(
                    model, train_loader, criterion, optimizer, epoch, epochs
                )
                
                # 验证阶段
                model.eval()
                val_loss = self._validate_epoch(
                    model, val_loader, criterion, epoch, epochs
                )
                
                # 更新学习率
                scheduler.step(val_loss)
                
                # 保存历史
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                
                # 检查是否需要保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    early_stopping_counter = 0
                    self._save_checkpoint(model, optimizer, epoch, val_loss)
                else:
                    early_stopping_counter += 1
                
                # 检查是否需要早停
                if early_stopping_counter >= patience:
                    self.logger.info(f"触发早停: {patience}轮未改善")
                    break
                    
        except KeyboardInterrupt:
            self.logger.warning("训练被用户中断")
            self.training_interrupted = True
        except Exception as e:
            self.logger.error(f"训练出错: {str(e)}")
            raise
        finally:
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                
        return model, self.best_val_loss
        
    def _train_epoch(self,
                    model: nn.Module,
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    total_epochs: int
                    ) -> float:
        """训练一个轮次
        
        Returns:
            平均训练损失
        """
        total_loss = 0
        for i, (states, moves) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, moves)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 每10批次打印一次进度
            if (i + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch [{epoch+1}/{total_epochs}] "
                    f"Batch [{i+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
                
        return total_loss / len(train_loader)
        
    def _validate_epoch(self,
                       model: nn.Module,
                       val_loader: DataLoader,
                       criterion: nn.Module,
                       epoch: int,
                       total_epochs: int
                       ) -> float:
        """验证一个轮次
        
        Returns:
            平均验证损失
        """
        total_loss = 0
        with torch.no_grad():
            for states, moves in val_loader:
                outputs = model(states)
                loss = criterion(outputs, moves)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(val_loader)
        self.logger.info(
            f"Epoch [{epoch+1}/{total_epochs}] "
            f"Validation Loss: {avg_loss:.4f}"
        )
        return avg_loss
        
    def _save_checkpoint(self,
                        model: nn.Module,
                        optimizer: optim.Optimizer,
                        epoch: int,
                        val_loss: float
                        ) -> None:
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        checkpoint_path = os.path.join(
            self.output_dir,
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"保存检查点到 {checkpoint_path}")
        
    def plot_training_history(self) -> None:
        """绘制训练历史曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"保存训练历史图表到 {plot_path}")
        
    def load_checkpoint(self,
                       model: nn.Module,
                       checkpoint_path: str
                       ) -> Tuple[nn.Module, Dict]:
        """加载检查点
        
        Returns:
            model: 加载了权重的模型
            checkpoint_info: 检查点信息
        """
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint
