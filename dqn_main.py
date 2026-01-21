import os
import random
import numpy as np
import gymnasium as gym
import gymnasium_env

from collections import deque
import csv

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


# ----------------- REPRODUCIBILITY -----------------
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic torch (more repeatable, slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- Replay Buffer -----------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ----------------- Model -----------------
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


def moving_average(x, window=200):
    x = np.array(x, dtype=np.float32)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window, dtype=np.float32) / window, mode="valid")


# ----------------- Single-seed train -----------------
def train_one_seed(
    seed: int,
    episodes: int,
    size: int,
    model_path: str,
    log_csv: str,
):
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Training seed={seed} on {device} ===", flush=True)

    # FAST settings (same as your fast setup)
    max_steps = 600
    train_every = 8
    batch_size = 64
    warmup = 800
    buffer_capacity = 60_000
    target_update_steps = 500

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9993

    env = gym.make(
        "gymnasium_env/GridWorld-v0",
        size=size,
        render_mode=None,
        max_steps=max_steps,
        show_path=False,
    )
    env.action_space.seed(seed)

    obs0, _ = env.reset(seed=seed)
    obs_dim = obs0.size
    n_actions = env.action_space.n

    policy_net = DQN(obs_dim, n_actions, hidden=128).to(device)
    target_net = DQN(obs_dim, n_actions, hidden=128).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    buffer = ReplayBuffer(capacity=buffer_capacity)

    # logs
    rewards = np.zeros(episodes, dtype=np.float32)
    success = np.zeros(episodes, dtype=np.float32)
    static_hits = np.zeros(episodes, dtype=np.float32)

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "success", "epsilon", "static_collisions"])

    step_count = 0
    success_window = deque(maxlen=100)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)  # deterministic per episode
        s_vec = flatten_obs(obs)

        terminated = truncated = False
        ep_reward = 0.0
        ep_hits = 0

        while not (terminated or truncated):
            step_count += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(s_vec).unsqueeze(0).to(device)
                    q = policy_net(s_t)[0]
                    action = int(torch.argmax(q).item())

            obs2, reward, terminated, truncated, info = env.step(action)

            if info.get("hit_static", False):
                ep_hits += 1

            s2_vec = flatten_obs(obs2)
            done = float(terminated or truncated)

            buffer.push(s_vec, action, reward, s2_vec, done)
            s_vec = s2_vec
            ep_reward += reward

            if (step_count % train_every == 0) and len(buffer) >= max(warmup, batch_size):
                s_b, a_b, r_b, s2_b, done_b = buffer.sample(batch_size)

                s_b = torch.from_numpy(s_b).to(device)
                a_b = torch.from_numpy(a_b).to(device).unsqueeze(1)
                r_b = torch.from_numpy(r_b).to(device)
                s2_b = torch.from_numpy(s2_b).to(device)
                done_b = torch.from_numpy(done_b).to(device)

                q_sa = policy_net(s_b).gather(1, a_b).squeeze(1)

                with torch.no_grad():
                    max_next_q = target_net(s2_b).max(dim=1).values
                    target = r_b + gamma * max_next_q * (1.0 - done_b)

                loss = loss_fn(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optimizer.step()

            if step_count % target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # record
        rewards[ep] = ep_reward
        success[ep] = 1.0 if terminated else 0.0
        static_hits[ep] = float(ep_hits)

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        success_window.append(int(success[ep]))

        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, rewards[ep], success[ep], epsilon, static_hits[ep]])

        if (ep + 1) % 200 == 0:
            sr = sum(success_window) / len(success_window)
            print(
                f"seed {seed} | ep {ep+1}/{episodes} | R {ep_reward:.1f} | eps {epsilon:.3f} | succ100 {sr:.2f} | hit {ep_hits}",
                flush=True
            )

    env.close()
    torch.save(policy_net.state_dict(), model_path)
    print(f"✅ Saved model: {model_path}", flush=True)

    return rewards, success, static_hits


# ----------------- Multi-seed runner + plots -----------------
def train_multi_seed(
    seeds=(1, 2, 3),
    episodes=8000,
    size=30,
    window=200,
):
    all_rewards = []
    all_success = []
    all_hits = []

    for s in seeds:
        model_path = f"fast_dqn_static_30_seed{s}.pt"
        log_csv = f"training_log_seed{s}.csv"
        r, succ, hits = train_one_seed(
            seed=s,
            episodes=episodes,
            size=size,
            model_path=model_path,
            log_csv=log_csv,
        )
        all_rewards.append(r)
        all_success.append(succ)
        all_hits.append(hits)

    R = np.stack(all_rewards, axis=0)     # (num_seeds, episodes)
    S = np.stack(all_success, axis=0)
    H = np.stack(all_hits, axis=0)

    # moving average (same length for all)
    def ma_stack(X):
        mas = []
        for i in range(X.shape[0]):
            mas.append(moving_average(X[i], window=window))
        # pad to same length already (convolution valid gives same length for each seed)
        return np.stack(mas, axis=0)

    Rm = ma_stack(R)
    Sm = ma_stack(S)
    Hm = ma_stack(H)

    # mean and std
    R_mean, R_std = Rm.mean(axis=0), Rm.std(axis=0)
    S_mean, S_std = Sm.mean(axis=0), Sm.std(axis=0)
    H_mean, H_std = Hm.mean(axis=0), Hm.std(axis=0)

    x = np.arange(len(R_mean)) + window  # episode index aligned

    # Reward plot
    plt.figure()
    plt.plot(x, R_mean)
    plt.fill_between(x, R_mean - R_std, R_mean + R_std, alpha=0.2)
    plt.title(f"Reward (mean±std over seeds, moving avg window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("learning_reward_mean_std.png", dpi=200)
    plt.close()

    # Success plot
    plt.figure()
    plt.plot(x, S_mean)
    plt.fill_between(x, S_mean - S_std, S_mean + S_std, alpha=0.2)
    plt.title(f"Success Rate (mean±std over seeds, moving avg window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("learning_success_mean_std.png", dpi=200)
    plt.close()

    # Collisions plot
    plt.figure()
    plt.plot(x, H_mean)
    plt.fill_between(x, H_mean - H_std, H_mean + H_std, alpha=0.2)
    plt.title(f"Static Collisions (mean±std over seeds, moving avg window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Static collisions / episode")
    plt.grid(True)
    plt.savefig("learning_static_collisions_mean_std.png", dpi=200)
    plt.close()

    print("\n📈 Saved mean±std plots:")
    print(" - learning_reward_mean_std.png")
    print(" - learning_success_mean_std.png")
    print(" - learning_static_collisions_mean_std.png")


if __name__ == "__main__":
    train_multi_seed(
        seeds=(1, 2, 3),
        episodes=8000,
        size=30,
        window=200,  # smoother than 100, better for reports
    )
