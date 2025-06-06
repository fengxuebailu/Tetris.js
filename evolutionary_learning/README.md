# 俄罗斯方块进化算法系统 (2025年版)

本项目是俄罗斯方块AI系统中的进化算法实现部分。通过模拟自然选择和进化过程，系统能够自动优化和进化出高效的游戏策略。2025年版本引入了更多优化算法和自适应进化策略。

## 系统特点

- **自动优化**: 无需人类训练数据，通过自我对弈和进化实现持续优化
- **高效性能**: 并行化的进化过程，支持多核心优化
- **稳定策略**: 经过进化的策略具有较强的稳定性和鲁棒性
- **易于扩展**: 模块化设计，便于添加新的进化策略和评估方法

## 项目结构

```
evolutionary_learning/
├── core/             # 进化算法核心实现
│   ├── tetris_evolution.py   # 进化算法主要逻辑
│   └── Tetris.py            # 游戏环境
├── models/           # 训练好的模型
│   └── best_weights.json    # 最佳权重配置
├── configs/          # 配置文件
│   ├── evolution_params.json # 进化参数配置
│   └── game_config.json     # 游戏配置
└── scripts/          # 工具脚本
    ├── start.py            # 主启动脚本
    ├── train.py           # 训练脚本
    └── evaluate.py        # 评估脚本
```

## 监督学习系统

监督学习系统通过观察人类玩家的游戏数据来训练AI模型。

### 主要功能

1. 训练新模型
2. 测试模型性能
3. 诊断模型问题
4. 评估所有模型

### 使用方法

```bash
cd supervised_learning
python scripts/start.py --train    # 训练新模型
python scripts/start.py --test     # 测试模型
python scripts/start.py --diagnose # 诊断模型
python scripts/start.py --evaluate # 评估所有模型
```

## 进化算法系统

进化算法系统使用遗传算法来优化AI的决策参数。

### 主要功能

1. 训练新模型
2. 测试模型性能
3. 评估所有模型

### 使用方法

```bash
cd evolutionary_learning
python scripts/start.py --train    # 训练新模型
python scripts/start.py --test     # 测试模型
python scripts/start.py --evaluate # 评估所有模型
```

## 历史文件说明

以下文件已不再使用，但保留作为参考：

### configs/ 目录
- `best_weights_evolved.json` - 旧版权重文件，已由models目录下的文件替代
- `old_params/` - 历史参数配置，仅供参考

### scripts/ 目录
- `play_with_evolved_weights.py` - 旧版演示脚本，请使用新版 `start.py`
- `best_weights_evolved.json` - 重复的权重文件，请使用configs目录下的配置

### core/ 目录
- `__pycache__/` - Python缓存文件，可以安全删除
- `old_evolution_strategies/` - 旧版进化策略实现

注意：所有带有 `.old`、`.bak` 或 `.deprecated` 后缀的文件均为历史版本，可以安全删除。
