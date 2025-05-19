# 俄罗斯方块监督学习系统

基于神经网络的俄罗斯方块AI训练和评估系统。

## 文件说明

### 核心文件
- `Tetris.py` - 俄罗斯方块游戏核心逻辑
- `tetris_supervised.py` - 原始监督学习系统（带有语法错误的版本）
- `tetris_supervised_fixed.py` - 修复版的监督学习系统
- `tetris_evolution.py` - 使用进化算法优化权重的系统

### 训练和测试文件
- `train_new_model.py` - 简单的新模型训练脚本
- `train_full_model.py` - 全面的模型训练脚本，支持完整数据集
- `test_models.py` - 基本的模型测试脚本

### 分析和可视化文件
- `analyze_models.py` - 深度模型性能分析脚本
- `visualize_model.py` - 神经网络模型可视化脚本
- `model_compatibility.py` - 处理不同模型架构兼容性的工具

### 主流程
- `run_full_process.py` - 完整流程执行脚本，包括训练、测试和分析

## 使用方法

### 训练新模型
```bash
python train_full_model.py
```
这将使用完整的训练数据集训练新的增强架构模型。

### 测试模型
```bash
python test_models.py tetris_model.pth tetris_model_new_full.pth
```
这将测试并比较两个模型的性能。

### 分析模型
```bash
python analyze_models.py tetris_model.pth tetris_model_new_full.pth
```
这将对模型进行更深入的分析，包括生成性能分布和游戏终止原因等可视化结果。

### 可视化模型
```bash
python visualize_model.py tetris_model_new_full.pth
```
这将生成模型权重、激活值和决策热图等可视化图表。

### 执行完整流程
```bash
python run_full_process.py
```
这将执行完整的训练、测试和分析流程。可以使用以下参数：
- `--skip-train`: 跳过训练阶段
- `--skip-test`: 跳过测试阶段
- `--skip-analysis`: 跳过分析阶段
- `--epochs 100`: 设置训练轮数（默认为50）
- `--games 50`: 设置测试游戏数量（默认为20）

## 神经网络架构

新的TetrisNet架构采用分离式设计，单独处理棋盘和方块特征：

```python
class TetrisNet(nn.Module):
    def __init__(self):
        super(TetrisNet, self).__init__()
        
        # 棋盘特征提取网络
        self.board_features = nn.Sequential(
            nn.Linear(200, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
        # 方块特征提取网络
        self.piece_features = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16)
        )
        
        # 组合网络 (用于最终决策)
        self.combined_network = nn.Sequential(
            nn.Linear(144, 128),  # 128 + 16 = 144
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
```

## 运行环境要求
- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- Seaborn (用于可视化)
