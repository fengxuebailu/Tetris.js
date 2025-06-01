import torch
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import math # Added for math.exp

# Adjust path to import from src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # This is reinforcement_learning/
src_path = os.path.join(parent_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

project_root_path = os.path.dirname(parent_dir) # This is Tetris.js/
if project_root_path not in sys.path:
    sys.path.append(project_root_path) # To import TetrisNet for model loading if needed by agent

from rl_environment import TetrisEnv
from rl_agent import DQNAgent
# from network_models import NUM_ACTIONS # NUM_ACTIONS is used by DQNAgent internally

def plot_durations(episode_durations, show_result=False, save_path=None):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration (Number of Steps)')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if save_path:
        plt.savefig(save_path)
    # if is_ipython:
    #     if not show_result:
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #     else:
    #         display.display(plt.gcf())

def main():
    # --- Configuration ---
    NUM_EPISODES = 500 # Number of training episodes
    REPLAY_CAPACITY = 10000
    BATCH_SIZE = 128
    GAMMA = 0.99 # Discount factor
    EPS_START = 0.9 # Epsilon-greedy starting value
    EPS_END = 0.05 # Epsilon-greedy ending value
    EPS_DECAY = 2000 # Epsilon decay rate (higher means slower decay) # MODIFIED FROM 1000 back to 2000
    TAU = 0.005 # Target network update rate (for soft updates, not used with hard updates)
    LR = 3e-4 # Learning rate for AdamW optimizer # MODIFIED FROM 1e-4
    TARGET_UPDATE_FREQ = 10 # Update target network every N episodes
    MODEL_SAVE_FREQ = 50 # Save model every N episodes
    LOG_FREQ = 1 # Log progress every N episodes

    # Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(parent_dir, "models", f"dqn_{timestamp}")
    log_dir = os.path.join(parent_dir, "logs", f"dqn_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    plot_save_path = os.path.join(log_dir, "training_durations.png")

    # Path to the pre-trained supervised model
    # supervised_learning/models/tetris_model_best.pth
    supervised_model_path = os.path.join(
        project_root_path,
        "supervised_learning",
        "models",
        "tetris_model_best.pth"
    )
    if not os.path.exists(supervised_model_path):
        print(f"Warning: Supervised model not found at {supervised_model_path}. Agent will start with random weights for feature extractor.")
        pretrained_feature_extractor_path = None
    else:
        pretrained_feature_extractor_path = supervised_model_path
        print(f"Found supervised model for pretraining at: {pretrained_feature_extractor_path}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialization ---
    env = TetrisEnv()
    # state_dim from env, action_dim from env or network_models
    # state_dim = env.observation_space.shape[0] # Assuming env has this, otherwise calculate from _get_state()
    # For TetrisEnv, state is a list, convert to numpy array then get shape
    
    # Modified to handle tuple unpacking from env.reset()
    initial_state_raw, initial_valid_actions_mask = env.reset()
    state_dim = np.array(initial_state_raw).shape[0]
    action_dim = env.action_space_n # 40 possible actions

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay=EPS_DECAY,
        tau=TAU, # Not used if hard updates are used
        lr=LR,
        device=device,
        pretrained_model_path=pretrained_feature_extractor_path
    )

    episode_durations = []
    total_steps = 0
    
    print(f"Starting training for {NUM_EPISODES} episodes...")

    # --- Training Loop ---
    for i_episode in range(NUM_EPISODES):
        # Modified to handle tuple unpacking from env.reset()
        state, valid_actions_mask = env.reset()
        
        current_episode_steps = 0
        episode_reward = 0
        done = False
        
        # --- MODIFICATION START: Detailed logging for specific episodes ---
        # log_this_episode_in_detail = (i_episode < 5) # Log first 5 episodes step-by-step
        # if log_this_episode_in_detail:
        #     print(f"--- Episode {i_episode + 1} Start (Detailed Log) ---")
        # --- MODIFICATION END ---

        while not done:
            # Pass the valid_actions_mask to the agent
            action_tensor = agent.select_action(state, valid_actions_mask) 
            action = action_tensor.item() # Get integer action

            # Modified to handle tuple unpacking from env.step()
            next_state_raw, reward, done, info, next_valid_actions_mask = env.step(action)
            
            # --- MODIFICATION START: Step-by-step logging ---
            # if log_this_episode_in_detail:
            #     print(f"  [Ep {i_episode+1} Step {current_episode_steps+1}] Action: {action}, Reward: {reward:.2f}, Done: {done}, Info: {info}")
            # --- MODIFICATION END ---
            
            episode_reward += reward
            current_episode_steps += 1
            total_steps +=1

            agent.store_transition(state, action, next_state_raw if not done else None, reward, done)

            state = next_state_raw
            valid_actions_mask = next_valid_actions_mask # Update mask for the next state

            loss = agent.optimize_model() # optimize_model handles train/eval mode switching

            if done:
                episode_durations.append(current_episode_steps)
                # --- MODIFICATION START: Log if episode was very short ---
                # if not log_this_episode_in_detail and current_episode_steps < 10 : # Log if episode was short (e.g. < 10 steps) and not already logged in detail
                #     print(f"--- Episode {i_episode + 1} (Short Episode Log, Steps: {current_episode_steps}) ---")
                # if log_this_episode_in_detail:
                #      print(f"--- Episode {i_episode + 1} End (Detailed Log). Total Reward: {episode_reward:.2f}, Steps: {current_episode_steps} ---")
                # --- MODIFICATION END ---
                break
        
        if (i_episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network_hard()
            # print(f"Episode {i_episode+1}: Target network updated.") # Reduced verbosity

        if (i_episode + 1) % LOG_FREQ == 0:
            avg_loss_str = "N/A" # Renamed to avoid conflict
            if loss is not None: 
                 avg_loss_str = f"{loss:.4f}" 
            
            current_epsilon = agent.eps_end + (agent.eps_start - agent.eps_end) * \
                              math.exp(-1. * agent.steps_done / agent.eps_decay)
            
            # Calculate average reward for the last LOG_FREQ episodes
            # Ensure there are enough elements in episode_durations for slicing
            if len(episode_durations) >= LOG_FREQ:
                # Assuming episode_reward is stored or accessible for past LOG_FREQ episodes
                # For simplicity, let's use episode_reward of the current episode for now,
                # or you would need to store all episode_rewards in a list like episode_durations
                # A better approach would be to store all episode_rewards:
                # all_episode_rewards.append(episode_reward)
                # avg_reward_recent = np.mean(all_episode_rewards[-LOG_FREQ:])
                # For now, using current episode_reward as a proxy for "recent"
                avg_reward_recent_placeholder = episode_reward # Placeholder
            else:
                avg_reward_recent_placeholder = episode_reward # Placeholder

            print(f"Episode {i_episode+1}/{NUM_EPISODES} | Steps: {current_episode_steps} | Ep Reward: {episode_reward:.2f} | Epsilon: {current_epsilon:.4f} | Loss: {avg_loss_str}")
            # Removed avg_reward_recent_placeholder from print to avoid confusion, as it's not a true average yet.
            # Consider adding a list to store all episode rewards for a proper moving average.
            
            if len(episode_durations) > 0: # Ensure plot_durations is called only if there's data
                plot_durations(episode_durations, save_path=None) 

        if (i_episode + 1) % MODEL_SAVE_FREQ == 0:
            model_path = os.path.join(model_dir, f"tetris_dqn_episode_{i_episode+1}.pth")
            agent.save_model(model_path)
            plot_durations(episode_durations, save_path=plot_save_path) # Save plot when model is saved
            print(f"Model saved at {model_path}")

    print("Training complete.")
    final_model_path = os.path.join(model_dir, "tetris_dqn_final.pth")
    agent.save_model(final_model_path)
    print(f"Final model saved at {final_model_path}")
    
    plot_durations(episode_durations, show_result=True, save_path=plot_save_path)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    # Check if display is available for plotting
    # try:
    #     from IPython import get_ipython
    #     if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
    #         from IPython import display
    #         is_ipython = True
    #         print("Running in IPython environment, plots will be interactive.")
    #     else:
    #         is_ipython = False
    #         print("Not in IPython, plots will be static or use plt.show().")
    # except ImportError:
    #     is_ipython = False
    #     print("IPython not found, plots will be static or use plt.show().")
    
    # For matplotlib GUI backend
    if os.environ.get('DISPLAY','') == '':
        print('No display found. Using non-interactive Agg backend for matplotlib.')
        plt.switch_backend('Agg')
    else:
        print(f"Display found: {os.environ.get('DISPLAY')}. Using default backend for matplotlib.")


    main()
