import torch
import numpy as np
import time
import os
from datetime import datetime
from matplotlib import pyplot as plt

from distributed.distributed import C51Distribution
from networks.network import C51NoisyDuelingCNN, C51NoisyDuelingMLP
from replay_buffer.per import NStepPrioritizedReplayBuffer
from envs.envs import make_env

import torch.optim as optim


class RainbowDQNAgent:
    # =========================
    # Initialization & Setup
    # =========================
    def __init__(self, config):

        # ---- Optimization ----
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.lr = config["lr"]
        self.tau = config["tau"]

        # ---- Environment / Architecture ----
        self.env_id = config["env_name"]
        self.net_type = config["network"]

        # ---- Replay (PER + n-step) ----
        self.memory_size = config["memory_size"]
        self.n_step = config["n_step"]

        # ---- C51 Distribution ----
        self.v_min = config["v_min"]
        self.v_max = config["v_max"]
        self.atom_size = config["atom_size"]

        # ---- Target network updates ----
        self.target_update_type = config.get("target_update_type", "hard")
        self.target_update_freq = config.get("target_update_freq", 5000)

        # ---- Internal counters ----
        self.learn_steps = 0
        self.solved = False

        # ---- Environment ----
        self.env = make_env(self.env_id)
        self.is_atari = self.env_id.startswith("ALE/")
        self.obs_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        if self.net_type == "cnn" and not self.is_atari:
            raise ValueError("CNN requires Atari environment")
        if self.net_type == "mlp" and self.is_atari:
            raise ValueError("MLP incompatible with Atari environment")

        # ---- Device ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Algorithm ----
        self._setup_algorithm()

        # ---- Replay buffer (always PER + n-step) ----
        self.memory = NStepPrioritizedReplayBuffer(
            capacity=self.memory_size,
            obs_shape=self.obs_shape,
            n_step=self.n_step,
            gamma=self.gamma
        )

        # ---- Logging ----
        self.reward_history = []
        self.run_dir = None
        self.run_name = None

    def _setup_algorithm(self):
        # Distribution
        self.c51 = C51Distribution(
            v_min=self.v_min,
            v_max=self.v_max,
            atom_size=self.atom_size,
            device=self.device
        )

        # Network
        if self.net_type == "cnn":
            net_cls = C51NoisyDuelingCNN
            kwargs = dict(
                num_actions=self.num_actions,
                atom_size=self.atom_size,
                support=self.c51.support
            )
        else:
            net_cls = C51NoisyDuelingMLP
            kwargs = dict(
                obs_dim=self.obs_shape[0],
                num_actions=self.num_actions,
                atom_size=self.atom_size,
                support=self.c51.support
            )

        self.policy_net = net_cls(**kwargs).to(self.device)
        self.target_net = net_cls(**kwargs).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    # =========================
    # Acting
    # =========================
    def select_action(self, state):
        self.policy_net.reset_noise()

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            dist = self.policy_net(state_t)
            q_vals = self.c51.expectation(dist)
            return q_vals.argmax(1).item()

    # =========================
    # Training Loop
    # =========================
    def train(self, max_frames=5_000_000, save_interval=30,
              live_plot_enabled=False, verbose=False, run_name="default"):

        self.run_name = run_name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join("runs", f"run-{run_name}-{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        last_save_time = time.time()
        save_interval *= 60

        frame = 0
        episode_reward = 0

        state, _ = self.env.reset()
        state = np.asarray(state, dtype=np.float32)

        while frame < max_frames:
            action = self.select_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32)
            done = terminated or truncated

            frame += 1
            episode_reward += reward

            self.memory.store(state, action, reward, next_state, done)

            if len(self.memory) >= self.batch_size:
                self.train_step()
                self.learn_steps += 1

                if self.target_update_type == "soft":
                    self.soft_update()
                elif self.target_update_type == "hard":
                    if self.learn_steps % self.target_update_freq == 0:
                        self.hard_update()

            if done:
                self.reward_history.append(episode_reward)

                if verbose:
                    ep = len(self.reward_history)
                    avg_100 = np.mean(self.reward_history[-50:])
                    print(
                        f"Episode {ep} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Avg Reward (100): {avg_100:.2f} | "
                    )
                
                if ep >= 50 and avg_100 >= 200.0:
                    print("Solved! Saving final model and stopping training.")

                    self.solved = True

                    ts = int(time.time())
                    self.save_model(os.path.join(self.run_dir, f"model_solved_{ts}.pth"))
                    np.save(
                        os.path.join(self.run_dir, f"rewards_solved_{ts}.npy"),
                        np.array(self.reward_history),
                    )

                    break

                episode_reward = 0
                state, _ = self.env.reset()
                state = np.asarray(state, dtype=np.float32)
            else:
                state = next_state

            if time.time() - last_save_time >= save_interval:
                ts = int(time.time())
                self.save_model(os.path.join(self.run_dir, f"model_{ts}.pth"))
                last_save_time = time.time()

            if live_plot_enabled and frame % 5000 == 0:
                self.plot_rewards_live()

    # =========================
    # Learning
    # =========================
    def train_step(self):
        states, actions, rewards, next_states, dones, gammas, idxs, weights = \
            self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        gammas = torch.tensor(gammas, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        dist = self.policy_net(states)
        dist_a = dist[range(states.size(0)), actions.squeeze()]

        with torch.no_grad():
            next_dist = self.policy_net(next_states)
            next_q = self.c51.expectation(next_dist)
            next_actions = next_q.argmax(dim=1)

            next_target_dist = self.target_net(next_states)
            next_dist_a = next_target_dist[range(states.size(0)), next_actions]

            target_dist = self.c51.project(next_dist_a, rewards, dones, gammas)

        loss_elem = -(target_dist * (dist_a + 1e-8).log()).sum(dim=1)
        loss = (loss_elem * weights.squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.memory.update_priorities(
            idxs, loss_elem.detach().cpu().numpy()
        )

    # =========================
    # Target updates
    # =========================
    def hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self):
        for name, param in self.policy_net.state_dict().items():
            self.target_net.state_dict()[name].copy_(
                self.tau * param + (1 - self.tau) * self.target_net.state_dict()[name]
            )

    # =========================
    # Utilities
    # =========================
    def plot_rewards_live(self):
        plt.clf()
        plt.plot(self.reward_history)
        plt.pause(0.01)

    def save_model(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "reward_history": self.reward_history,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "reward_history" in checkpoint:
            self.reward_history = checkpoint["reward_history"]

        self.policy_net.eval()
        self.target_net.eval()

        print(f"[LOAD] Model loaded from {path}")

