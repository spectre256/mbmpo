import numpy as np
import torch
import gymnasium as gym
import time
from main import Policy  # Import the exact Policy class from your main script

def evaluate_agent(model_path="mb_mpo_policy.pt", num_episodes=5, render=True):
    # 1. Initialize environment
    render_mode = "human" if render else None
    env = gym.make("Ant-v5", render_mode=render_mode)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # 2. Reconstruct Policy Architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(obs_dim, act_dim).to(device)
    
    # 3. Load the Weights
    print(f"Loading weights from {model_path}...")
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()  # Put the policy in evaluation mode
    
    # 4. Run Evaluation Loops
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # Deterministic evaluation step (use mu/loc instead of sampling if preferred)
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                dist = policy.dist(obs_t)
                action = dist.loc.cpu().numpy()  # Using mean action for clean test execution
                
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            obs = next_obs
            done = terminated or truncated
            
            if render:
                time.sleep(0.01)  # Slows down rendering slightly so you can watch the ant
                
        print(f"Episode {ep+1} Finished | Total Steps: {steps} | Total Reward: {ep_reward:.2f}")
        
    env.close()

if __name__ == "__main__":
    # Run a test loop. Set render=False if you are working over an SSH connection without a display window.
    evaluate_agent(model_path="mb_mpo_policy.pt", num_episodes=3, render=True)