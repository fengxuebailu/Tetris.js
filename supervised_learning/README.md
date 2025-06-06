# 俄罗斯方块监督学习系统 (2025年版)

本目录包含了基于监督学习的俄罗斯方块AI系统的完整实现。该系统通过分析和学习人类玩家的游戏数据来训练AI模型，实现了一个能够模仿并超越人类玩家策略的智能体。通过持续的改进和优化，系统在2025年的版本中加入了多项新功能和优化。

## 系统特点

- **数据驱动**: 基于真实玩家数据进行训练
- **多种网络架构**: 支持标准、改进和健壮三种不同的网络结构
- **完整工具链**: 包含训练、测试、诊断和评估等全套工具
- **可视化支持**: 提供详细的训练过程和结果可视化
- **模型诊断**: 内置强大的诊断工具，帮助识别和解决问题

## 目录结构

### core/ - 核心实现
- `Tetris.py` - 俄罗斯方块游戏核心逻辑
- `tetris_supervised.py` - 基础监督学习实现
- `tetris_supervised_fixed.py` - 改进版监督学习实现
- `enhanced_training.py` - 增强版训练系统
- `train_robust_model.py` - 健壮型模型训练器
- `training_pipeline.py` - 完整训练流水线

### tools/ - 工具集
- `test_models.py` - 模型测试工具
- `comprehensive_evaluate.py` - 全面评估工具
- `diagnose_invalid_moves.py` - 无效移动诊断工具
- `analyze_models.py` - 模型分析工具
- `evaluate_models.py` - 模型评估工具
- `visualize_model.py` - 可视化工具
- `model_compatibility.py` - 模型兼容性处理

### models/ - 模型文件
- `tetris_model.pth` - 基础模型
- `tetris_model_best.pth` - 最佳性能模型
- `tetris_model_new_full.pth` - 新架构完整训练模型
- 其他epoch保存点模型文件

### data/ - 数据文件
- `tetris_training_data.npz` - 训练数据集

### configs/ - 配置文件
- `best_weights_evolved.json` - 进化算法得到的最佳权重

### scripts/ - 脚本文件
- `setup.ps1` - Windows环境设置脚本
- `setup.sh` - Unix环境设置脚本
- `start.py` - 启动脚本

### 输出目录
- `move_diagnostics/` - 诊断报告和可视化结果
- `training_logs/` - 训练日志和历史记录

## 快速开始

1. 环境设置：
   - Windows:
   ```powershell
   .\scripts\setup.ps1
   ```
   - Linux/MacOS:
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. 使用启动脚本：
   ```bash
   python scripts/start.py --help
   ```

## 主要功能

1. 训练新模型：
   ```bash
   python scripts/start.py --train
   ```

2. 测试模型：
   ```bash
   python scripts/start.py --test
   ```

3. 诊断无效移动：
   ```bash
   python scripts/start.py --diagnose
   ```

4. 全面评估：
   ```bash
   python scripts/start.py --evaluate
   ```

## 系统要求

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib
- Seaborn (用于可视化)

## 模型架构

系统包含三种主要的模型架构：

1. **标准架构** (TetrisNet)
   - 简单的前馈神经网络
   - 适合基础训练和快速实验

2. **改进架构** (ImprovedTetrisNet)
   - 包含残差连接和批归一化
   - 更深的网络结构
   - 分离的特征提取网络

3. **健壮架构** (RobustTetrisNet)
   - 专注于避免无效移动
   - 多级验证和调整策略
   - 增强的边界情况处理

## 调试和优化

如果遇到问题：

1. 检查 `move_diagnostics` 目录中的诊断报告
2. 查看 `training_logs` 目录中的训练历史
3. 使用 `visualize_model.py` 分析模型行为
4. 通过 `comprehensive_evaluate.py` 进行全面评估

## 注意事项

1. 训练前请确保有足够的训练数据
2. 使用 `--validate` 选项进行模型验证
3. 定期检查无效移动率
4. 保存重要的模型检查点

## 历史文件说明

以下文件已不再使用，但保留作为参考：

### core/ 目录
- `game_engine.py` - 已由 `game_engine_cv.py` 替代
- `train_full_model.py` - 已由 `training_pipeline.py` 替代

### tools/ 目录
- `old_test_framework/` - 旧版测试框架，已由新版替代
- `legacy_visualizer.py` - 旧版可视化工具，建议使用 `visualize_model.py`

### training_data/ 目录
- `old_training_samples/` - 旧版训练数据，已由新数据集替代
- `deprecated_features.txt` - 弃用特征列表，仅供参考

### evolutionary_learning/ 目录
此目录为历史遗留，已迁移至项目根目录的evolutionary_learning模块。
