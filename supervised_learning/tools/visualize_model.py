#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 神经网络可视化脚本
用于可视化和分析模型内部结构、权重和激活
"""

# 导入Matplotlib配置
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tools.matplotlibrc import *

import torch
import numpy as np
import seaborn as sns
from core.tetris_supervised_fixed import TetrisNet, TetrisAI
from Tetris import shapes, rotate, check, join_matrix, clear_rows

def visualize_model_structure(model_path):
    """可视化模型结构"""
    try:
        # 加载模型
        model = TetrisNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # 打印模型结构
        print(f"\n=== {os.path.basename(model_path)} 模型结构 ===")
        print(model)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        return model
    
    except Exception as e:
        print(f"可视化模型结构时出错: {e}")
        return None

def visualize_model_weights(model):
    """可视化模型权重"""
    if model is None:
        return
    
    try:
        # 创建目录
        if not os.path.exists("model_visualization"):
            os.makedirs("model_visualization")
        
        # 可视化每一层的权重分布
        plt.figure(figsize=(15, 10))
        layer_idx = 0
        
        # 函数来绘制一个模块中的权重
        def plot_module_weights(module, prefix="", max_plots=9):
            nonlocal layer_idx
            for name, param in module.named_parameters():
                if 'weight' in name:
                    layer_idx += 1
                    if layer_idx <= max_plots:
                        plt.subplot(3, 3, layer_idx)
                        weights = param.data.cpu().numpy().flatten()
                        sns.histplot(weights, bins=50, kde=True)
                        plt.title(f"{prefix}{name}")
                        plt.xlabel("权重值")
                        plt.ylabel("频率")
        
        # 可视化主要模块的权重
        plot_module_weights(model.board_features, "board_features.")
        plot_module_weights(model.piece_features, "piece_features.")
        plot_module_weights(model.combined_network, "combined_network.")
        
        plt.tight_layout()
        plt.savefig("model_visualization/weight_distributions.png")
        
        # 可视化第一层的权重
        plt.figure(figsize=(12, 5))
        
        # 棋盘特征第一层
        plt.subplot(1, 2, 1)
        board_weights = model.board_features[0].weight.data.cpu().numpy()
        plt.imshow(board_weights, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title("棋盘特征提取第一层权重")
        plt.xlabel("输入特征")
        plt.ylabel("神经元")
        
        # 方块特征第一层
        plt.subplot(1, 2, 2)
        piece_weights = model.piece_features[0].weight.data.cpu().numpy()
        plt.imshow(piece_weights, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title("方块特征提取第一层权重")
        plt.xlabel("输入特征")
        plt.ylabel("神经元")
        
        plt.tight_layout()
        plt.savefig("model_visualization/first_layer_weights.png")
        
    except Exception as e:
        print(f"可视化模型权重时出错: {e}")

def visualize_activations(model_path):
    """可视化模型激活值"""
    try:
        # 加载AI和模型
        ai = TetrisAI(model_path)
        
        # 创建一个空游戏板和一个示例方块
        board = [[0 for _ in range(10)] for _ in range(20)]
        piece = shapes[0]  # I形方块
        
        # 获取状态向量
        state_vector = ai.create_state_vector(board, piece)
        state_tensor = torch.FloatTensor([state_vector])
        
        # 获取模型各部分的激活值
        board_input = state_tensor[:, :200]
        piece_input = state_tensor[:, 200:]
        
        # 获取中间激活
        model = ai.model
        board_features = model.board_features(board_input).detach().cpu().numpy()[0]
        piece_features = model.piece_features(piece_input).detach().cpu().numpy()[0]
        
        # 可视化激活值
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(board_features)), board_features)
        plt.title("棋盘特征提取激活值")
        plt.xlabel("特征索引")
        plt.ylabel("激活值")
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(piece_features)), piece_features)
        plt.title("方块特征提取激活值")
        plt.xlabel("特征索引")
        plt.ylabel("激活值")
        
        plt.tight_layout()
        plt.savefig("model_visualization/activations.png")
        
    except Exception as e:
        print(f"可视化激活值时出错: {e}")

def visualize_decision_heatmap(model_path):
    """生成决策热图"""
    try:
        # 加载AI
        ai = TetrisAI(model_path)
        
        # 创建一个带有一些方块的游戏板
        board = [[0 for _ in range(10)] for _ in range(20)]
        
        # 底部添加一些方块
        for j in range(10):
            board[19][j] = 1
        
        for j in range(8):
            board[18][j] = 1
        
        # 为I形方块创建决策热图
        piece = shapes[0]  # I形方块
        
        # 为不同的x位置和旋转创建热图数据
        heatmap_data = np.zeros((4, 10))  # 4种旋转，10种可能的x位置
        
        # 生成所有可能旋转的方块
        rotated_pieces = [piece]
        for _ in range(3):
            rotated_pieces.append(rotate(rotated_pieces[-1]))
        
        # 填充热图
        for rot in range(4):
            rotated_piece = rotated_pieces[rot]
            for x in range(-2, 8):
                # 尝试放置方块
                piece_width = len(rotated_piece[0])
                if x + piece_width > 10:
                    continue
                    
                # 计算分数
                y = 0
                while y < 20 and check(board, rotated_piece, [x, y+1]):
                    y += 1
                
                # 检查是否可以放置
                if check(board, rotated_piece, [x, y]):
                    # 模拟放置并评估
                    temp_board = [row[:] for row in board]
                    join_matrix(temp_board, rotated_piece, [x, y])
                    new_board, cleared = clear_rows(temp_board)
                    
                    # 使用模型预测此位置的分数
                    state_vector = ai.create_state_vector(board, rotated_piece)
                    state_tensor = torch.FloatTensor([state_vector])
                    
                    with torch.no_grad():
                        prediction = ai.model(state_tensor).cpu().numpy()[0]
                        # 使用预测值作为分数
                        score = abs(prediction[0] - x) + abs(prediction[1] - rot)
                        # 映射到0-1范围
                        score = 1.0 / (1.0 + score)
                        
                        # 填入热图
                        x_idx = x + 2  # 调整索引，因为x可以是负数
                        if 0 <= x_idx < 10:
                            heatmap_data[rot, x_idx] = score
        
        # 可视化热图
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f")
        plt.title(f"{os.path.basename(model_path)}: I形方块决策热图")
        plt.xlabel("X位置")
        plt.ylabel("旋转")
        plt.tight_layout()
        plt.savefig("model_visualization/decision_heatmap.png")
        
    except Exception as e:
        print(f"生成决策热图时出错: {e}")
        import traceback
        traceback.print_exc()

def compare_model_outputs(model_paths):
    """比较不同模型的输出"""
    models = []
    model_names = []
    
    # 加载所有模型
    for path in model_paths:
        try:
            ai = TetrisAI(path)
            models.append(ai)
            model_names.append(os.path.basename(path))
        except Exception as e:
            print(f"加载模型 {path} 失败: {e}")
    
    if len(models) < 2:
        print("至少需要两个模型来进行比较")
        return
    
    # 创建一些测试场景
    test_scenarios = []
    
    # 场景1：空棋盘
    board1 = [[0 for _ in range(10)] for _ in range(20)]
    test_scenarios.append(("空棋盘", board1))
    
    # 场景2：底部填满
    board2 = [[0 for _ in range(10)] for _ in range(20)]
    for j in range(10):
        board2[19][j] = 1
    test_scenarios.append(("底部填满", board2))
    
    # 场景3：有一个洞
    board3 = [[0 for _ in range(10)] for _ in range(20)]
    for j in range(10):
        if j != 5:
            board3[19][j] = 1
    test_scenarios.append(("底部有一个洞", board3))
    
    # 比较每个场景的输出
    print("\n=== 不同模型输出比较 ===")
    
    for scenario_name, board in test_scenarios:
        print(f"\n场景: {scenario_name}")
        
        for piece_idx, piece in enumerate(shapes[:3]):  # 测试前三种方块
            print(f"方块类型: {piece_idx}")
            
            for i, ai in enumerate(models):
                move = ai.predict_move(board, piece)
                print(f"- {model_names[i]}: x={move['x']}, 旋转={move['rotation']}")

# 导入需要的函数
from Tetris import rotate, check, join_matrix, clear_rows

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python visualize_model.py <模型路径>")
        print("示例: python visualize_model.py tetris_model.pth")
        
        # 使用最新的pth文件
        import os
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        
        if not model_files:
            print("错误: 找不到任何模型文件")
            sys.exit(1)
            
        # 按修改时间排序，选择最新的
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        model_path = model_files[0]
        print(f"使用最新的模型文件: {model_path}")
    else:
        model_path = sys.argv[1]
    
    # 确保可视化目录存在
    if not os.path.exists("model_visualization"):
        os.makedirs("model_visualization")
    
    # 可视化模型结构
    print(f"可视化模型: {model_path}")
    model = visualize_model_structure(model_path)
    
    # 可视化模型权重
    print("可视化模型权重...")
    visualize_model_weights(model)
    
    # 可视化模型激活值
    print("可视化模型激活值...")
    visualize_activations(model_path)
    
    # 生成决策热图
    print("生成决策热图...")
    visualize_decision_heatmap(model_path)
    
    # 如果提供了多个模型，比较它们的输出
    if len(sys.argv) > 2:
        compare_model_outputs(sys.argv[1:])
    elif len([f for f in os.listdir('.') if f.endswith('.pth')]) > 1:
        # 使用最新的两个模型
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        compare_model_outputs(model_files[:2])
        
    print("\n可视化完成。结果保存在 model_visualization 目录中。")
