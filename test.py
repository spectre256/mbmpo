import numpy as np
import torch
import gymnasium as gym
import imageio
import os
from main import Policy

def evaluate_agent(model_path="mb_mpo_policy.pt", video_path="eval_videos/ant_wiggle.mp4"):
    env = gym.make(
        "Ant-v5",
        render_mode="rgb_array",
        width=1920,
        height=1080
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(obs_dim, act_dim).to(device)

    print(f"Loading weights from {model_path}")
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    video_writer = imageio.get_writer(video_path, fps=50, codec='libx264', quality=10)

    # 500 total steps at 50 FPS = 10 seconds of simulation data
    TARGET_RECORDING_STEPS = 500
    total_recorded_steps = 0
    episode_count = 0

    try:
        while total_recorded_steps < TARGET_RECORDING_STEPS:
            obs, _ = env.reset()
            ep_reward = 0.0
            steps = 0
            done = False
            episode_count += 1

            while not done and total_recorded_steps < TARGET_RECORDING_STEPS:
                # Fast evaluation forward pass (using mean action)
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                    dist = policy.dist(obs_t)
                    action = dist.loc.cpu().numpy()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                steps += 1
                obs = next_obs
                done = terminated or truncated

                frame = env.render()
                if frame is not None:
                    video_writer.append_data(frame)
                    total_recorded_steps += 1

            print(f"Episode {episode_count}, step {steps}, progress: {total_recorded_steps}/{TARGET_RECORDING_STEPS}")

    except KeyboardInterrupt:
        print("\nRecording interrupted")
    finally:
        video_writer.close()
        env.close()
        print(f"Video saved to {video_path}")

if __name__ == "__main__":
    evaluate_agent(model_path="mb_mpo_policy.pt")
