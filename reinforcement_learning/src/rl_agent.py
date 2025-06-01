import math
import random
import numpy as np
from collections import namedtuple, deque
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 从当前目录的 network_models.py 导入 DQNNet
# 假设 rl_agent.py 和 network_models.py 在同一个 src/ 目录下
# from .network_models import DQNNet, NUM_ACTIONS # . 表示当前包
from network_models import DQNNet, NUM_ACTIONS # Changed to direct import

# 定义经验的结构
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存一个经验"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """随机采样一批经验"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self,
                 state_dim, 
                 action_dim=NUM_ACTIONS, 
                 hidden_dims=512, # Kept for potential future use, but not used by current DQNNet
                 replay_capacity=10000,
                 batch_size=128,
                 gamma=0.99,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=1000, 
                 tau=0.005, 
                 lr=1e-4,
                 device=None,
                 pretrained_model_path=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.hidden_dims = hidden_dims # Not directly used by DQNNet constructor
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # print(f"Using device: {self.device}") # Already printed in test script

        # Initialize Q网络 (策略网络和目标网络)
        # DQNNet constructor in network_models.py only takes num_actions
        self.policy_net = DQNNet(num_actions=self.action_dim).to(self.device)
        self.target_net = DQNNet(num_actions=self.action_dim).to(self.device)

        if pretrained_model_path:
            try:
                # 假设 DQNNet 有一个 load_pretrained_feature_extractor 方法
                self.policy_net.load_pretrained_feature_extractor(pretrained_model_path)
                print(f"Successfully loaded pretrained weights into policy_net from {pretrained_model_path}")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}. Starting with random weights for feature extractor.")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval() # Set to eval mode by default
        self.target_net.eval() # Target network is always in eval mode

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(replay_capacity)

        self.steps_done = 0

    def select_action(self, state, valid_actions_mask=None): # Added valid_actions_mask
        """根据epsilon-greedy策略选择动作，并考虑动作掩码"""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold: # 利用: 选择Q值最大的动作
            with torch.no_grad():
                self.policy_net.eval()
                if not isinstance(state, torch.Tensor):
                    state_tensor = torch.tensor(np.array([state]), device=self.device, dtype=torch.float32)
                else:
                    # If state is already a tensor, ensure it has batch dimension
                    state_tensor = state.unsqueeze(0) if state.ndim == 1 else state 
                state_tensor = state_tensor.to(self.device)


                q_values = self.policy_net(state_tensor) # Shape (1, action_dim)

                if valid_actions_mask is not None:
                    q_values_masked = q_values.clone()
                    # Ensure mask is a boolean tensor on the same device
                    mask_tensor_bool = torch.tensor(valid_actions_mask, device=self.device, dtype=torch.bool)
                    
                    if not mask_tensor_bool.any(): # If all actions are invalid by the mask
                        print("Warning: All actions masked in select_action (exploit). Choosing a random action from all.")
                        return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
                    
                    # Apply mask: set Q-values of invalid actions to -infinity
                    q_values_masked[0, ~mask_tensor_bool] = -float('inf') 
                    action = q_values_masked.max(1)[1].view(1, 1)
                else: # No mask provided, old behavior
                    action = q_values.max(1)[1].view(1, 1)
                return action
        else: # 探索: 随机选择一个动作
            if valid_actions_mask is not None:
                valid_indices = [i for i, valid in enumerate(valid_actions_mask) if valid]
                if valid_indices:
                    chosen_action = random.choice(valid_indices)
                    return torch.tensor([[chosen_action]], device=self.device, dtype=torch.long)
                else:
                    # Fallback if all actions are masked
                    print("Warning: All actions masked in select_action (explore). Choosing a random action from all.")
                    return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
            else: # No mask provided, old behavior
                return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def store_transition(self, state, action, next_state, reward, done):
        """将经验存储到回放缓冲区"""
        # 将所有数据转换为tensor
        state = torch.tensor(np.array([state]), device=self.device, dtype=torch.float32)
        action = torch.tensor([[action]], device=self.device, dtype=torch.long) if not isinstance(action, torch.Tensor) else action.to(self.device)

        if next_state is not None:
            next_state = torch.tensor(np.array([next_state]), device=self.device, dtype=torch.float32)
        else: # 如果是终止状态，next_state 可以是 None
            next_state = None

        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        done = torch.tensor([done], device=self.device, dtype=torch.bool)

        self.memory.push(state, action, next_state, reward, done)

    def optimize_model(self):
        """从经验回放中采样并优化模型"""
        if len(self.memory) < self.batch_size:
            return None # Not enough samples in memory

        self.policy_net.train() # Set policy_net to train mode for optimization

        transitions = self.memory.sample(self.batch_size)
        # 将一批经验转换为一个 Transition 对象，其中每个字段都是一个包含所有样本的张量
        # (e.g., batch.state 是一个包含所有状态的张量)
        batch = Transition(*zip(*transitions))

        # 计算非终止状态的掩码，并连接批处理元素
        # (最终模型优化部分需要这个)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                     device=self.device, dtype=torch.bool)

        # 处理 batch.next_state 中的 None 值
        # 创建一个包含所有非 None next_state 的张量
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.cat(non_final_next_states_list).to(self.device)
        else:
            # 如果所有 next_state 都是 None (所有都是终止状态)
            non_final_next_states = torch.empty((0, self.state_dim), device=self.device, dtype=torch.float32)


        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        # done_batch = torch.cat(batch.done).to(self.device) # done_batch is not directly used in Q calculation like this

        # 计算 Q(s_t, a) - 模型计算 Q(s_t)，然后我们选择采取的动作的列。
        # 这些是根据 policy_net 对每个批处理状态所采取的操作。
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算所有下一个状态的 V(s_{t+1})。
        # 非最终状态的下一个状态的预期操作值是基于“较旧的” target_net 计算的；
        # 用 max(1)[0] 选择它们最佳的奖励。这是基于贝尔曼方程。
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states.size(0) > 0: # 确保有非终止的下一状态
            with torch.no_grad(): # 不计算梯度，因为这是目标网络
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # 计算期望的 Q 值 (TD Target)
        # expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # 修正：如果状态是终止状态 (done=True)，则其期望Q值仅为奖励
        # done_mask_for_q_value = (~torch.cat(batch.done)).float() # 1 if not done, 0 if done
        # expected_state_action_values = reward_batch + (self.gamma * next_state_values * done_mask_for_q_value)
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        # Ensure expected_state_action_values has the same shape as state_action_values
        expected_state_action_values = expected_state_action_values.unsqueeze(1)


        # 计算损失 (例如 Huber 损失)
        criterion = nn.SmoothL1Loss() # Huber loss
        loss = criterion(state_action_values, expected_state_action_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping (可选，但有助于稳定训练)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.policy_net.eval() # Return policy_net to eval mode

        return loss.item()

    def update_target_network_hard(self):
        """硬更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Policy network saved to {path}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 同步目标网络
        self.policy_net.eval() # 通常加载后用于评估
        self.target_net.eval()
        print(f"Policy network loaded from {path}")

# 简单的测试代码 (可选)
if __name__ == '__main__':
    print("Testing DQNAgent...")
    # Parameters
    INPUT_DIMS = 219 
    NUM_ACTIONS_TEST = 40 # Renamed to avoid conflict with NUM_ACTIONS from network_models
    HIDDEN_DIMS_TEST = 512 

    current_script_path = os.path.dirname(os.path.abspath(__file__)) 
    rl_folder_path = os.path.dirname(current_script_path) 
    project_root_path = os.path.dirname(rl_folder_path) 
    
    supervised_model_path = os.path.join(
        project_root_path,
        "supervised_learning",
        "models",
        "tetris_model_best.pth"
    )
    print(f"Pretrained model for agent test found at: {supervised_model_path}")
    
    if not os.path.exists(supervised_model_path):
        print(f"Error: Supervised model not found at {supervised_model_path}")
        # exit(1) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize agent
    agent = DQNAgent(
        state_dim=INPUT_DIMS,
        action_dim=NUM_ACTIONS_TEST,
        hidden_dims=HIDDEN_DIMS_TEST, # Passed but not used by DQNNet constructor
        pretrained_model_path=supervised_model_path,
        device=device,
        eps_decay=500 
    )

    print("Testing action selection...")
    dummy_state = np.random.rand(INPUT_DIMS).tolist()

    for i in range(5): 
        action_tensor = agent.select_action(dummy_state) # select_action returns a tensor
        action_item = action_tensor.item() # Get the integer value
        print(f"Step {i+1}, Epsilon: {agent.eps_end + (agent.eps_start - agent.eps_end) * math.exp(-1. * agent.steps_done / agent.eps_decay):.4f}, Action: {action_item}")

    print("\\nTesting experience replay and optimization (dummy data)...")
    for _ in range(200): 
        dummy_next_state = np.random.rand(INPUT_DIMS).tolist()
        # Ensure action is an int for storage, select_action now returns a tensor
        dummy_action_tensor = agent.select_action(dummy_state) 
        dummy_action_item = dummy_action_tensor.item()
        dummy_reward = random.random()
        dummy_done = random.choice([True, False])
        # Store the integer action
        agent.store_transition(dummy_state, dummy_action_item, dummy_next_state, dummy_reward, dummy_done)
        dummy_state = dummy_next_state if not dummy_done else np.random.rand(INPUT_DIMS).tolist()
    
    print(f"Memory size: {len(agent.memory)}")

    print("Attempting one optimization step...")
    loss = agent.optimize_model()
    if loss is not None:
        print(f"Optimization step successful. Loss: {loss:.4f}")
    else:
        print("Optimization step skipped (not enough samples).")

    agent.update_target_network_hard() 
    print("\\nTarget network updated.")

    test_model_save_path = os.path.join(rl_folder_path, "models", "test_dqn_agent_model.pth")
    os.makedirs(os.path.join(rl_folder_path, "models"), exist_ok=True) 
    
    agent.save_model(test_model_save_path)
    print(f"Model saved to {test_model_save_path}")

    new_agent = DQNAgent(
        state_dim=INPUT_DIMS,
        action_dim=NUM_ACTIONS_TEST,
        hidden_dims=HIDDEN_DIMS_TEST,
        device=device,
        eps_start=0.0, # Force greedy for testing loaded model
        eps_end=0.0    # Force greedy for testing loaded model
    )
    new_agent.load_model(test_model_save_path)
    print(f"Model loaded into new agent from {test_model_save_path}")

    # select_action needs a list/numpy array, not a tensor for state
    # dummy_state was last updated to a list.
    action_loaded_tensor = new_agent.select_action(dummy_state) 
    action_loaded_item = action_loaded_tensor.item()
    print(f"Action from loaded model (greedy): {action_loaded_item}")
    
    if os.path.exists(test_model_save_path):
        os.remove(test_model_save_path)
        print(f"Cleaned up test model file: {test_model_save_path}")

    print("\\nDQNAgent test script finished.")

