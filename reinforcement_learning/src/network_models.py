import sys
import os

# 将项目根目录添加到 sys.path
# __file__ 是当前文件 (network_models.py) 的路径
# os.path.dirname(__file__) 是 src/ 目录
# os.path.dirname(os.path.dirname(__file__)) 是 reinforcement_learning/ 目录
# os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 是 Tetris.js/ (项目根目录)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn

# 尝试从监督学习模块导入 TetrisNet
# 这假设 'supervised_learning' 包在 Python 路径中，通常项目根目录会被添加到路径中。
try:
    from supervised_learning.core.tetris_supervised_fixed import TetrisNet
except ImportError as e:
    print(f"Error importing TetrisNet: {e}")
    print("Please ensure that the project root directory (Tetris.js) is in your PYTHONPATH,")
    print("or that the supervised_learning module is correctly installed/accessible.")
    # 作为后备，如果直接导入失败，可以考虑将TetrisNet定义复制到此处或使用更复杂的路径处理
    # For now, we'll raise the error if it cannot be imported, as it's crucial.
    raise

NUM_ACTIONS = 40  # 4 rotations * 10 x-positions

class DQNNet(nn.Module):
    def __init__(self, num_actions=NUM_ACTIONS):
        super(DQNNet, self).__init__()

        # 复制 TetrisNet 中的特征提取层结构
        # 1. 卷积层处理游戏板状态
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1))  # 输出大小为 64x1x1
        )

        # 2. 处理额外特征 (高度、空洞、凹凸度)
        self.extra_features = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1)
        )

        # 3. 处理当前方块特征
        self.piece_features = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32)
        )

        # Q值输出头
        # 合并后的特征维度: 64 (conv) + 32 (extra) + 32 (piece) = 128
        self.q_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        # 分离不同类型的输入，与 TetrisNet 一致
        # x 的形状: (batch_size, 219)
        # board_input: (batch_size, 200)
        # extra_input: (batch_size, 3)
        # piece_input: (batch_size, 16)
        board_input = x[:, :200]
        extra_input = x[:, 200:203]
        piece_input = x[:, 203:]

        # 处理游戏板 - 重塑为 2D 并添加通道维度
        # (batch_size, 200) -> (batch_size, 1, 20, 10)
        board_feats = board_input.view(-1, 1, 20, 10)
        board_feats = self.conv_layers(board_feats)  # 输出: (batch_size, 64, 1, 1)
        board_feats = board_feats.view(-1, 64)      # 展平: (batch_size, 64)

        # 处理额外特征和方块特征
        extra_feats = self.extra_features(extra_input) # 输出: (batch_size, 32)
        piece_feats = self.piece_features(piece_input) # 输出: (batch_size, 32)

        # 合并所有特征
        combined_feats = torch.cat([board_feats, extra_feats, piece_feats], dim=1) # 输出: (batch_size, 128)

        # 通过Q值头生成Q值
        q_values = self.q_head(combined_feats) # 输出: (batch_size, num_actions)
        return q_values

    def load_pretrained_feature_extractor(self, supervised_model_path):
        """
        从预训练的 TetrisNet 加载特征提取部分的权重。
        Args:
            supervised_model_path (str): 预训练的 TetrisNet 模型 (.pth) 文件路径。
        """
        try:
            print(f"Attempting to load supervised model weights from: {supervised_model_path}")
            # 创建一个临时的 TetrisNet 实例来加载完整的预训练模型
            temp_supervised_net = TetrisNet()
            
            # 加载预训练模型的 state_dict
            # 使用 map_location 确保在不同设备上（CPU/GPU）都能正确加载
            state_dict = torch.load(supervised_model_path, map_location=lambda storage, loc: storage)
            temp_supervised_net.load_state_dict(state_dict)
            print("Successfully loaded state_dict into temporary TetrisNet.")

            # 将预训练的特征提取层权重复制到当前模型的对应层
            self.conv_layers.load_state_dict(temp_supervised_net.conv_layers.state_dict())
            self.extra_features.load_state_dict(temp_supervised_net.extra_features.state_dict())
            self.piece_features.load_state_dict(temp_supervised_net.piece_features.state_dict())
            
            print("Successfully copied weights to DQNNet feature extractors.")
            
        except FileNotFoundError:
            print(f"Error: Supervised model weights file not found at {supervised_model_path}")
            print("Please ensure the path is correct and the file exists.")
        except AttributeError as e:
            print(f"Error: Could not find TetrisNet or its layers. Is TetrisNet imported correctly? Details: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while loading pretrained weights: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    # 这是一个简单的测试/示例代码，当直接运行此文件时执行
    print("Testing DQNNet creation and pretrained weight loading...")

    # 假设的输入状态维度 (batch_size, feature_dim)
    batch_size = 2
    input_features = 219  # 200 (board) + 3 (extra) + 16 (piece)
    dummy_input = torch.randn(batch_size, input_features)

    # 创建 DQNNet 实例
    dqn_model = DQNNet()
    print("DQNNet instance created.")

    # 打印模型结构
    # print(dqn_model)

    # 尝试加载预训练权重
    # 注意: 修改此路径为实际的预训练模型路径
    # 从 network_models.py (reinforcement_learning/src/) 到 supervised_learning/models/
    relative_path_to_supervised_model = os.path.join(
        os.path.dirname(__file__), # 当前文件 (src) 的目录
        '..',                     # reinforcement_learning/
        '..',                     # Tetris.js/ (项目根目录)
        'supervised_learning',
        'models',
        'tetris_model_best.pth'
    )
    absolute_path_to_supervised_model = os.path.abspath(relative_path_to_supervised_model)
    
    if os.path.exists(absolute_path_to_supervised_model):
        dqn_model.load_pretrained_feature_extractor(absolute_path_to_supervised_model)
    else:
        print(f"Could not find supervised model at: {absolute_path_to_supervised_model}")
        print("Skipping pretrained weight loading test.")

    # 测试前向传播
    try:
        q_values = dqn_model(dummy_input)
        print(f"Forward pass successful. Output Q-values shape: {q_values.shape}") # 应为 (batch_size, NUM_ACTIONS)
        assert q_values.shape == (batch_size, NUM_ACTIONS)
    except Exception as e:
        print(f"Error during forward pass: {e}")

    print("Test finished.")
