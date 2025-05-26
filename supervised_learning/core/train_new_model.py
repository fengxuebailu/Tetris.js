#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
俄罗斯方块监督学习系统 - 模型训练脚本
用于从现有数据训练新版本的模型
"""

from tetris_supervised import TetrisDataset, TetrisNet, train_network, save_model

def train_new_model():
    """从现有数据训练新模型"""
    print("=== 训练新版本模型 ===")
    
    try:
        # 加载训练数据
        print("加载训练数据...")
        game_states, moves = TetrisDataset.load_from_file()
        
        if game_states is None or moves is None or len(game_states) == 0:
            print("错误: 找不到训练数据或数据为空")
            return False
        
        print(f"成功加载训练数据: {len(game_states)} 个样本")
        print(f"数据形状: game_states={game_states.shape}, moves={moves.shape}")
        
        # 快速训练模式，使用少量数据进行测试
        print("\n开始训练新模型 (简化模式，仅10轮)...")
        
        # 创建更小的数据集用于快速测试
        sample_size = min(1000, len(game_states))
        print(f"使用 {sample_size} 个样本进行快速训练")
        
        test_states = game_states[:sample_size]
        test_moves = moves[:sample_size]
        
        model = train_network(test_states, test_moves, 
                            num_epochs=10,     # 仅训练10轮进行测试 
                            batch_size=32,     # 更小的批次
                            learning_rate=0.001,  # 学习率0.001
                            patience=5)       # 5轮没有提升就早停
                            
    except Exception as e:
        print(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 保存新模型
    new_model_name = "tetris_model_new_arch.pth"
    save_model(model, new_model_name)
    print(f"新模型已保存到: {new_model_name}")
    
    return model

if __name__ == "__main__":
    # 训练新模型
    new_model = train_new_model()
    
    # 测试新旧模型对比
    if new_model:
        print("\n测试新旧模型对比:")
        import sys
        sys.path.append('.')
        from model_compatibility import compare_models
        
        print("\n=== 模型对比测试 ===")
        old_model = compare_models("tetris_model.pth", "tetris_model_new_arch.pth")
