import gymnasium as gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cpu")


def set_seed(env, seed=42):
    # Reproducibility for fair hyperparam comparisons.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class ReplayMemory:
    def __init__(self, capacity):
        # Replay buffer reduces correlation in updates.
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Dueling + C51 (categorical distributional DQN)
class DuelingC51(nn.Module):
    def __init__(self, n_obs, n_actions, n_atoms):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Value stream V(s) over atoms (logits for N bins)
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_atoms),
        )

        # Advantage stream A(s,a) over atoms (logits for A*N bins)
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions * n_atoms),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)  # (B, N)
        a = self.advantage(f).view(-1, self.n_actions, self.n_atoms)  # (B, A, N)

        # Mean-normalised advantage to make decomposition identifiable
        logits = v.unsqueeze(1) + (a - a.mean(dim=1, keepdim=True))  # (B, A, N)

        # Softmax over atoms gives categorical distribution per action
        probs = F.softmax(logits, dim=2)  # (B, A, N)
        return logits, probs


def expected_q_from_probs(probs, support):
    # Recover scalar Q(s,a) by expectation over atom support.
    return (probs * support.view(1, 1, -1)).sum(dim=2)  # (B, A)


@torch.no_grad()
def c51_projection(next_probs, rewards, dones, gamma, support, vmin, vmax):
    # C51 projection onto fixed support.
    B, N = next_probs.shape
    delta_z = (vmax - vmin) / (N - 1)

    # Tz = r + gamma*(1-done)*z
    tz = rewards.unsqueeze(1) + gamma * (1.0 - dones.unsqueeze(1)) * support.view(1, -1)
    tz = tz.clamp(vmin, vmax)

    b = (tz - vmin) / delta_z  # (B, N)
    l = b.floor().long()
    u = b.ceil().long()

    l = l.clamp(0, N - 1)
    u = u.clamp(0, N - 1)

    m = torch.zeros_like(next_probs)  # (B, N)

    offset = (torch.arange(B, device=next_probs.device) * N).unsqueeze(1)  # (B, 1)
    l_idx = (l + offset).view(-1)
    u_idx = (u + offset).view(-1)

    # Split mass between neighbours (linear interpolation)
    m.view(-1).index_add_(0, l_idx, (next_probs * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, u_idx, (next_probs * (b - l.float())).view(-1))

    # EDGE CASE: if l == u, push all mass to that bin
    eq = (u == l).view(-1)
    if eq.any():
        eq_idx = l_idx[eq]
        m.view(-1).index_add_(0, eq_idx, next_probs.view(-1)[eq])

    return m


def moving_average(x, window=100):
    if len(x) < window:
        return [None] * len(x)
    out = [None] * (window - 1)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    out.extend(ma.tolist())
    return out


def plot_rewards(rewards, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.plot(rewards, label="Return")
    plt.plot(moving_average(rewards, 100), label="MA(100)")
    plt.legend()
    plt.show()


def train_c51_dueling(
    env_id,
    num_episodes,
    seed=42,
    batch_size=128,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=2000,
    lr=3e-4,
    replay_capacity=100000,
    max_steps_per_episode=2000,
    target_update=1000,
    learning_starts=1000,
    grad_clip_norm=10.0,
    # C51 params
    vmin=-200.0,
    vmax=200.0,
    n_atoms=51,
    double_c51=True,
):
    env = gym.make(env_id)
    set_seed(env, seed)

    n_actions = env.action_space.n
    n_obs = env.observation_space.shape[0]

    # Fixed atom support z_i in [vmin, vmax]
    support = torch.linspace(vmin, vmax, n_atoms, device=device)

    policy_net = DuelingC51(n_obs, n_actions, n_atoms).to(device)
    target_net = DuelingC51(n_obs, n_actions, n_atoms).to(device)

    # Start aligned so early targets aren’t random noise.
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(replay_capacity)

    steps_done = 0
    total_steps = 0
    episode_returns = []

    def select_action(state_np):
        # Classic epsilon-greedy with smooth step decay.
        nonlocal steps_done
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
        steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                state_t = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
                _, probs = policy_net(state_t)           # (1, A, N)
                q = expected_q_from_probs(probs, support) # (1, A)
                return q.argmax(dim=1).view(1, 1)

        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < batch_size:
            return

        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device)

        non_final_mask = torch.tensor([ns is not None for ns in batch.next_state],
                                      device=device, dtype=torch.bool)

        # Predicted distribution for chosen actions (log p_theta(z_i|s,a))
        logits, _ = policy_net(state_batch)                      # (B, A, N)
        log_probs = F.log_softmax(logits, dim=2)                 # log-softmax over atoms
        chosen_log_probs = log_probs[torch.arange(batch_size, device=device),
                                     action_batch.squeeze(1)]    # (B, N)

        # Terminal transitions: no bootstrap, so next dist is irrelevant.
        next_probs_full = torch.full((batch_size, n_atoms), 1.0 / n_atoms, device=device)

        if non_final_mask.any():
            non_final_next_states = torch.tensor(
                np.array([ns for ns in batch.next_state if ns is not None]),
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                # Target net provides bootstrap distribution
                _, t_probs = target_net(non_final_next_states)  # (Bnf, A, N)

                if double_c51:
                    # Pick a* with policy net, evaluate with target net.
                    _, p_probs = policy_net(non_final_next_states)
                    q_next = expected_q_from_probs(p_probs, support)
                    a_star = q_next.argmax(dim=1)
                else:
                    q_next = expected_q_from_probs(t_probs, support)
                    a_star = q_next.argmax(dim=1)

                next_probs = t_probs[torch.arange(t_probs.size(0), device=device), a_star]  # (Bnf, N)
                next_probs_full[non_final_mask] = next_probs

        # Project Bellman target distribution back onto fixed support
        target_dist = c51_projection(
            next_probs=next_probs_full,
            rewards=reward_batch,
            dones=done_batch,
            gamma=gamma,
            support=support,
            vmin=vmin,
            vmax=vmax,
        )  # (B, N)

        # Cross-entropy loss between target_dist and predicted distribution for the chosen action
        loss = -(target_dist * chosen_log_probs).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()

        # Keeps updates stable without changing the objective.
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip_norm)

        optimizer.step()

    for i_episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        ep_return = 0.0

        for _ in range(max_steps_per_episode):
            action_t = select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action_t.item())
            done = terminated or truncated

            total_steps += 1
            ep_return += float(reward)

            next_state = None if done else obs
            memory.push(state, action_t.item(), next_state, float(reward), done)
            state = obs

            # Warm-up avoids learning from tiny, highly correlated samples.
            if total_steps > learning_starts:
                optimize_model()

            # Hard target sync mirrors standard DQN.
            if total_steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_returns.append(ep_return)

        if i_episode % 25 == 0:
            last_25 = episode_returns[-25:]
            print(
                f"[{env_id}] Episode {i_episode}/{num_episodes} | "
                f"AvgReturn(last25)={np.mean(last_25):.2f}"
            )

    env.close()
    return policy_net, episode_returns, (vmin, vmax, n_atoms)


def evaluate_c51(policy_net, env_id, dist_params, n_episodes=20, seed=123, max_steps=2000):
    # Greedy evaluation to measure learnt policy.
    vmin, vmax, n_atoms = dist_params
    support = torch.linspace(vmin, vmax, n_atoms, device=device)

    env = gym.make(env_id)
    set_seed(env, seed)

    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_return = 0.0

        for _ in range(max_steps):
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                _, probs = policy_net(state_t)
                q = expected_q_from_probs(probs, support)
                action = q.argmax(dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            state = obs

            if terminated or truncated:
                break

        returns.append(ep_return)

    env.close()
    return returns


import os
import imageio.v2 as imageio


def record_gif_c51(policy_net, dist_params, env_id="LunarLander-v3", filename="agent_video",
                   seed=123, max_steps=2000, fps=30):
    vmin, vmax, n_atoms = dist_params
    support = torch.linspace(vmin, vmax, n_atoms, device=device)

    env = gym.make(env_id, render_mode="rgb_array")
    set_seed(env, seed)

    frames = []
    state, _ = env.reset()
    frames.append(env.render())

    for _ in range(max_steps):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            _, probs = policy_net(state_t)
            q = expected_q_from_probs(probs, support)
            action = q.argmax(dim=1).item()

        state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())

        if terminated or truncated:
            break

    env.close()

    gif_path = f"{filename}.gif"
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"GIF saved to: {os.path.abspath(gif_path)}")

    display(Image(filename=gif_path))


# # Run CartPole
# cartpole_net, cartpole_returns, cartpole_dist = train_c51_dueling(
#     env_id="CartPole-v1",
#     num_episodes=500,
#     seed=42,
#     replay_capacity=50000,
#     batch_size=64,
#     gamma=0.99,
#     eps_start=1.0,
#     eps_end=0.05,
#     eps_decay=2000,
#     lr=3e-4,
#     max_steps_per_episode=500,
#     target_update=500,
#     learning_starts=1000,
#     vmin=0.0,
#     vmax=500.0,
#     n_atoms=51,
#     double_c51=True,
# )

# plot_rewards(cartpole_returns, "CartPole-v1 | Dueling C51 (epsilon decay)")

# cartpole_eval = evaluate_c51(cartpole_net, "CartPole-v1", cartpole_dist, n_episodes=30, max_steps=500)
# print(
#     f"[CartPole-v1] Eval over 30 episodes | "
#     f"Mean={np.mean(cartpole_eval):.2f} | Std={np.std(cartpole_eval):.2f}"
# )

# Run LunarLander
lunar_net, lunar_returns, lunar_dist = train_c51_dueling(
    env_id="LunarLander-v3",
    num_episodes=1200,
    seed=42,
    replay_capacity=100000,
    batch_size=128,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.02,
    eps_decay=50000,
    lr=1e-4,
    max_steps_per_episode=2000,
    target_update=2000,
    learning_starts=10000,
    vmin=-300.0,
    vmax=300.0,
    n_atoms=51,
    double_c51=True,
)

plot_rewards(lunar_returns, "LunarLander-v3 | Dueling C51 (epsilon decay)")

record_gif_c51(lunar_net, lunar_dist, env_id="LunarLander-v3", filename="lunar_lander_video")

lunar_eval = evaluate_c51(lunar_net, "LunarLander-v3", lunar_dist, n_episodes=30, max_steps=2000)
print(
    f"[LunarLander-v3] Eval over 30 episodes | "
    f"Mean={np.mean(lunar_eval):.2f} | Std={np.std(lunar_eval):.2f}"
)
