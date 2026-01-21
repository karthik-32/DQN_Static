import time
import numpy as np
import gymnasium as gym
import gymnasium_env
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    return obs.reshape(-1).astype(np.float32)


def main():
    size = 30
    model_file = "fast_dqn_static_30.pt"
    max_steps = 600

    # ✅ show_path=True only for play
    env = gym.make(
        "gymnasium_env/GridWorld-v0",
        size=size,
        render_mode="human",
        max_steps=max_steps,
        show_path=True
    )

    obs0, _ = env.reset()
    obs_dim = obs0.size
    n_actions = env.action_space.n

    model = DQN(obs_dim, n_actions, hidden=128)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    obs, _ = env.reset()
    time.sleep(0.5)

    terminated = truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        steps += 1
        s = flatten_obs(obs)

        with torch.no_grad():
            q = model(torch.from_numpy(s).unsqueeze(0))[0].numpy()

        action = int(np.argmax(q))
        obs, reward, terminated, truncated, _ = env.step(action)

        time.sleep(0.06)

    print(f"Done. steps={steps}, terminated={terminated}, truncated={truncated}")
    print("Close window to exit.")
    while True:
        env.render()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
