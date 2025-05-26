# 俄罗斯方块AI系统

本项目包含两个独立的AI系统：监督学习系统和进化算法系统。两个系统共用基础的游戏逻辑代码。

## 项目结构

```
Tetris.js/
├── supervised_learning/    # 监督学习系统
│   ├── core/             # 核心实现
│   ├── models/           # 训练好的模型(.pth格式)
│   ├── tools/           # 工具脚本
│   ├── data/            # 训练数据
│   └── scripts/         # 启动脚本
│
├── evolutionary_learning/ # 进化算法系统
│   ├── core/            # 核心实现
│   ├── models/          # 训练好的模型(.json格式)
│   ├── configs/         # 配置文件
│   └── scripts/         # 启动脚本
│
└── Tetris.py           # 基础游戏逻辑
```

## 监督学习系统

监督学习系统通过观察人类玩家的游戏数据来训练AI模型。使用深度神经网络实现。

### 使用方法

```bash
cd supervised_learning/scripts
python start.py --train    # 训练新模型
python start.py --test     # 测试模型
python start.py --diagnose # 诊断模型
python start.py --evaluate # 评估所有模型
```

## 进化算法系统

进化算法系统使用遗传算法来优化AI的决策参数。通过不断进化和选择来找到最优的游戏策略。

### 使用方法

```bash
cd evolutionary_learning/scripts
python start.py --train    # 训练新模型
python start.py --test     # 测试模型
python start.py --evaluate # 评估所有模型
```

## 区别

1. 监督学习系统:
   - 使用神经网络模型
   - 需要人类玩家的训练数据
   - 模型文件格式为.pth
   - 可以学习复杂的策略模式

2. 进化算法系统:
   - 使用遗传算法优化权重参数
   - 不需要训练数据,通过自我对弈进化
   - 模型文件格式为.json
   - 策略相对简单但稳定
