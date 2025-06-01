\
import torch
import numpy as np
import os
import sys
import time

# Adjust path to import from src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # This is reinforcement_learning/
src_path = os.path.join(parent_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

project_root_path = os.path.dirname(os.path.dirname(parent_dir)) # This is Tetris.js/
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

from rl_environment import TetrisEnv
from rl_agent import DQNAgent

def play_episodes(agent, env, num_episodes=10, render=True, model_name="DQN"):
    """
    Plays a specified number of episodes using the trained agent.

    Args:
        agent (DQNAgent): The trained agent.
        env (TetrisEnv): The Tetris environment.
        num_episodes (int): Number of episodes to play.
        render (bool): Whether to render the game.
        model_name (str): Name of the model for printing.
    """
    total_scores = []
    total_steps = []

    print(f"\\nPlaying {num_episodes} episodes with {model_name}...")

    for i_episode in range(num_episodes):
        state = env.reset()
        episode_score = 0
        episode_steps = 0
        done = False

        while not done:
            # Agent selects action greedily (epsilon should be 0)
            action_tensor = agent.select_action(state)
            action = action_tensor.item()

            next_state, reward, done, info = env.step(action)
            
            episode_score += reward
            episode_steps += 1
            state = next_state

            if render:
                env.render()
                time.sleep(0.05)  # Small delay to make rendering viewable

            if done:
                total_scores.append(episode_score)
                total_steps.append(episode_steps)
                print(f"Episode {i_episode + 1}: Score = {episode_score:.2f}, Steps = {episode_steps}")
                if render:
                    print("Game Over. Press Enter to continue to the next episode...")
                    input() # Wait for user to see final state if rendering
                break
    
    avg_score = np.mean(total_scores)
    avg_steps = np.mean(total_steps)
    print(f"\\n--- {model_name} Performance ---")
    print(f"Average Score over {num_episodes} episodes: {avg_score:.2f}")
    print(f"Average Steps over {num_episodes} episodes: {avg_steps:.2f}")
    print(f"Max Score: {np.max(total_scores):.2f}, Min Score: {np.min(total_scores):.2f}")
    print(f"Max Steps: {np.max(total_steps)}, Min Steps: {np.min(total_steps)}")
    return avg_score, avg_steps

def main():
    # --- Configuration ---
    MODEL_PATH = os.path.join(parent_dir, "models", "dqn_20250601_125328", "tetris_dqn_final.pth") # Or choose a specific episode model
    # Example for a specific episode:
    # MODEL_PATH = os.path.join(parent_dir, "models", "dqn_20250601_125328", "tetris_dqn_episode_500.pth")
    
    NUM_PLAY_EPISODES = 10
    RENDER_GAME = True # Set to False for faster evaluation without visuals

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure the model path is correct and the model has been trained.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialization ---
    env = TetrisEnv()
    initial_state_example = env.reset() # To get state_dim
    state_dim = np.array(initial_state_example).shape[0]
    action_dim = env.action_space_n

    # Initialize agent - for playing, eps_start and eps_end should be 0 for greedy actions
    # pretrained_model_path is not used here as we load the full agent state dict
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        eps_start=0.0,  # Greedy policy for evaluation
        eps_end=0.0,    # Greedy policy for evaluation
        device=device
    )

    try:
        agent.load_model(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Play episodes
    play_episodes(agent, env, num_episodes=NUM_PLAY_EPISODES, render=RENDER_GAME, model_name="Trained DQN")

    if RENDER_GAME:
        env.close() # Close the Pygame window if it was opened

if __name__ == '__main__':
    main()
