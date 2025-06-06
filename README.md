# 俄罗斯方块AI系统

本项目是一个综合性的俄罗斯方块AI系统，包含三个独立的AI实现方法：监督学习系统、进化算法系统和强化学习系统。这些系统共用基础的游戏逻辑代码，但采用不同的方法来训练和优化AI模型。

## 快速开始指南

### 1. 环境准备

```powershell
# 克隆项目
git clone https://github.com/your_username/Tetris.js.git
cd Tetris.js

# 安装全局依赖
pip install -r requirements.txt
```

### 2. 选择系统并安装

监督学习系统:

```powershell
cd supervised_learning
# Windows环境
.\setup.ps1
# 或Linux/Mac环境
./setup.sh
```

进化算法系统:

```powershell
cd evolutionary_learning
pip install -r requirements.txt
```

强化学习系统:

```powershell
cd reinforcement_learning
pip install -r requirements.txt
```

### 3. 启动程序

每个子系统都使用其scripts目录下的启动脚本作为入口。

监督学习系统:

```powershell
cd supervised_learning/scripts
python start.py --train    # 训练新模型
python start.py --test     # 测试模型
python start.py --diagnose # 诊断模型问题
python start.py --evaluate # 评估所有模型
```

进化算法系统:

```powershell
cd evolutionary_learning/scripts
python start.py --train    # 训练新模型
python start.py --test     # 测试模型
python start.py --evaluate # 评估模型
```

强化学习系统:

```powershell
cd reinforcement_learning/scripts
python train_rl_agent.py   # 训练新模型
python play_rl_agent.py    # 运行游戏演示
```

### 4. 输出文件位置

* 监督学习系统的模型保存在 `supervised_learning/models/`
* 进化算法系统的权重保存在 `evolutionary_learning/models/`
* 强化学习系统的模型保存在 `reinforcement_learning/models/`
* 游戏截图保存在 `game_screenshots/` 目录

### 5. 开发建议

* 使用 `scripts/` 目录下的启动脚本进行日常操作
* 核心算法在各系统的 `core/` 或 `src/` 目录中
* 配置文件位于各系统的 `configs/` 目录
* 使用 `tools/` 目录中的工具进行调试和分析

## 系统特点

- **多样化的AI方法**: 实现了三种不同的AI训练方法，可以比较它们的优劣
- **模块化设计**: 各个系统相互独立，便于维护和扩展
- **共享基础**: 所有系统共用相同的游戏逻辑，确保结果可比性

## 项目结构

```
Tetris.js/
├── supervised_learning/     # 监督学习系统
│   ├── core/              # 核心实现
│   ├── models/            # 训练好的模型(.pth格式)
│   ├── tools/            # 工具脚本
│   ├── data/             # 训练数据
│   └── scripts/          # 启动脚本
│
├── evolutionary_learning/  # 进化算法系统
│   ├── core/             # 核心实现
│   ├── models/           # 训练好的模型(.json格式)
│   ├── configs/          # 配置文件
│   └── scripts/          # 启动脚本
│
├── reinforcement_learning/ # 强化学习系统
│   ├── src/              # 核心源代码
│   ├── models/           # 训练好的模型(.pth格式)
│   ├── scripts/          # 工具脚本
│   └── logs/             # 训练日志
│
├── game_screenshots/      # 游戏截图
│   └── ...               # 各场次游戏记录
│
├── Tetris.py             # 基础游戏逻辑
│
└── [历史文件]            # 以下为历史开发文件，保留但不再使用
    ├── best_weights_evolved.json   # 旧版进化权重文件
    ├── run_full_process.py        # 旧版训练流程
    ├── test_debug_fixed.py        # 旧版调试脚本
    ├── test_debug.py              # 旧版调试脚本
    ├── test_simple.py             # 旧版测试脚本
    ├── test_supervised.py         # 旧版监督学习测试
    └── train_supervised.py        # 旧版训练脚本
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

## 系统比较

1. **监督学习系统**:
   - 使用神经网络模型，需要人类玩家数据
   - 适合模仿人类玩家的策略
   - 平均分数：10000+
   - 优势：稳定可靠，策略接近人类

2. **进化算法系统**:
   - 使用遗传算法优化参数，通过自我对弈进化
   - 适合发现创新性策略
   - 平均分数：12000+
   - 优势：不需要训练数据，策略独特

3. **强化学习系统**:
   - 使用深度强化学习，通过环境交互学习
   - 适合长期策略优化
   - 平均分数：15000+
   - 优势：性能最优，可持续进步

## 快速开始指南

1. **环境准备**:

   ```powershell
   # 克隆项目
   git clone https://github.com/your_username/Tetris.js.git
   cd Tetris.js
   
   # 安装全局依赖
   pip install -r requirements.txt
   ```

2. **系统选择与安装**:

   a) 监督学习系统:
   ```powershell
   cd supervised_learning
   # Windows环境
   .\setup.ps1
   # 或Linux/Mac环境
   ./setup.sh
   ```

   b) 进化算法系统:
   ```powershell
   cd evolutionary_learning
   pip install -r requirements.txt
   ```

   c) 强化学习系统:
   ```powershell
   cd reinforcement_learning
   pip install -r requirements.txt
   ```

3. **启动程序**:

   每个子系统都使用其scripts目录下的start.py作为入口：

   a) 监督学习系统:
   ```powershell
   cd supervised_learning/scripts
   python start.py --train    # 训练新模型
   python start.py --test     # 测试模型
   python start.py --diagnose # 诊断模型问题
   python start.py --evaluate # 评估所有模型
   ```

   b) 进化算法系统:
   ```powershell
   cd evolutionary_learning/scripts
   python start.py --train    # 训练新模型
   python start.py --test     # 测试模型
   python start.py --evaluate # 评估模型
   ```

   c) 强化学习系统:
   ```powershell
   cd reinforcement_learning/scripts
   python train_rl_agent.py   # 训练新模型
   python play_rl_agent.py    # 运行游戏演示
   ```

4. **查看结果**:
   - 监督学习系统的模型保存在 `supervised_learning/models/`
   - 进化算法系统的权重保存在 `evolutionary_learning/models/`
   - 强化学习系统的模型保存在 `reinforcement_learning/models/`
   - 游戏截图保存在 `game_screenshots/` 目录

5. **开发建议**:
   - 使用 `scripts/` 目录下的启动脚本进行日常操作
   - 核心算法在各系统的 `core/` 或 `src/` 目录中
   - 配置文件位于各系统的 `configs/` 目录
   - 使用 `tools/` 目录中的工具进行调试和分析

## 常见问题解决

1. **如果遇到模块导入错误**:
   ```powershell
   # 添加项目根目录到PYTHONPATH
   $env:PYTHONPATH = "当前目录路径;$env:PYTHONPATH"
   ```

2. **如果需要清理训练数据**:
   ```powershell
   # 清理缓存和临时文件
   Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
   Remove-Item "training_logs/*" -Recurse -Force
   ```

3. **如果需要重置环境**:
   ```powershell
   # 重新安装依赖
   pip install -r requirements.txt --force-reinstall
   ```

## 系统要求

- Python 3.8+
- PyTorch 1.8+
- NumPy
- TensorFlow (可选，用于部分功能)
- CUDA支持 (推荐，用于GPU加速)

## 注意事项

1. **关于历史文件**
   - 项目根目录下存在一些早期开发阶段的文件
   - 这些文件已被更好的实现所替代，但被保留用于参考
   - 建议使用各子系统目录下的最新实现
   - 历史文件列表：
     * `best_weights_evolved.json`: 由新版进化算法配置替代
     * `run_full_process.py`: 由各子系统的训练脚本替代
     * `test_*.py`系列文件: 由新版测试框架替代
     * `train_supervised.py`: 由监督学习系统的训练流程替代

2. **版本兼容性**
   - 历史文件可能与当前环境不完全兼容
   - 如需参考旧实现，请查看git历史中的相应版本
   - 推荐使用各子系统目录下的最新实现
