# 俄罗斯方块AI系统

本项目包含两个主要部分：监督学习系统和进化算法系统。

## 目录结构

```
supervised_learning/    # 监督学习系统
├── core/              # 核心实现
├── models/            # 训练好的模型
├── tools/            # 工具脚本
├── data/             # 训练数据
└── scripts/          # 启动脚本

evolutionary_learning/ # 进化算法系统
├── core/             # 核心实现
├── models/           # 训练好的模型
├── configs/          # 配置文件
└── scripts/          # 启动脚本
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
