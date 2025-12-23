import torch
import numpy as np
import time
import os
from datetime import datetime
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.optim as optim

from replay_buffer.replay_standard import ReplayBuffer
from exploration.exploration import LinearEpsilonScheduler
from envs.envs import make_env
from networks.DuelingMLP import DuelingMLP



class DDQNAgent:
    # =========================
    # Initialization & Setup
    # =========================
    def __init__(self, config, seed=5):

        self.seed = seed

        # ---- Optimization ----
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.lr = config["lr"]
        # self.tau = config["tau"]

        # ---- Exploration (ε-greedy) ----
        self.eps_start = config["eps_start"]
        self.eps_end = config["eps_end"]
        self.eps_decay = config["eps_decay"]

        self.epsilon = LinearEpsilonScheduler(
            eps_start=self.eps_start,
            eps_end=self.eps_end,
            decay_steps=self.eps_decay
        )

        # ---- Environment / Architecture ----
        self.env_id = config["env_name"]

        # ---- Replay (PER + n-step) ----
        self.memory_size = config["memory_size"]
        self.n_step = config["n_step"]

        # ---- Target network updates ----
        self.target_update_type = config.get("target_update_type", "hard")
        self.target_update_freq = config.get("target_update_freq", 1000)

        # ---- Internal counters ----
        self.learn_steps = 0
        self.solved = False

        # ---- Environment ----
        self.env = make_env(self.env_id)
        self.obs_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        # ---- Device ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Networks ----
        self._setup_networks()

        # ---- Replay buffer ----
        self.memory = ReplayBuffer(
            obs_dim=self.obs_shape[0],
            size=self.memory_size,
            batch_size=self.batch_size
        )

        # ---- Logging ----
        self.reward_history = []
        self.total_frames = 0

        self.run_dir = None
        self.run_name = None

    def _setup_networks(self):
        obs_dim = self.obs_shape[0]

        self.policy_net = DuelingMLP(
            obs_dim=obs_dim,
            num_actions=self.num_actions
        ).to(self.device)

        self.target_net = DuelingMLP(
            obs_dim=obs_dim,
            num_actions=self.num_actions
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    # =========================
    # Acting
    # =========================
    def select_action(self, state):
        if np.random.rand() < self.epsilon.value():
            return np.random.randint(self.num_actions)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state_t).argmax(1).item()
        
    # =========================
    # Training Loop
    # =========================
    def train(self, max_episodes=500, max_frames=1_000_000, save_interval=30,
              live_plot_enabled=False, verbose=False, run_name="default"):

        self.run_name = run_name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join("runs", f"run-{run_name}-{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        last_save_time = time.time()
        save_interval *= 60

        frame = 0
        episode_reward = 0
        episode = 0
        final_saved = False


        state, _ = self.env.reset(seed=self.seed)
        state = np.asarray(state, dtype=np.float32)

        while frame < max_frames:
            action = self.select_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = np.asarray(next_state, dtype=np.float32)
            done = terminated or truncated

            frame += 1
            self.total_frames += 1
            episode_reward += reward

            self.memory.store(state, action, reward, next_state, done)

            if len(self.memory) >= self.batch_size:
                self.train_step()
                self.learn_steps += 1
                self.epsilon.step()

                if self.target_update_type == "soft":
                    self.soft_update()
                elif self.target_update_type == "hard":
                    if self.learn_steps % self.target_update_freq == 0:
                        self.hard_update()

            if done:
                episode += 1
                self.reward_history.append({
                    "episode": len(self.reward_history) + 1,
                    "frames": self.total_frames,
                    "return": episode_reward
                })

                
                ep = len(self.reward_history)
                last_returns = [x["return"] for x in self.reward_history[-20:]]
                avg_100 = np.mean(last_returns)
                print(
                    f"Episode {ep} | "
                    f"frames {self.total_frames} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg Reward (100): {avg_100:.2f} | "
                )

                # -------- SOLVED CONDITION --------
                if episode >= 20 and avg_100 >= 200.0:
                    print("Solved! Saving final model and stopping training.")

                    ts = int(time.time())
                    self.save_model(os.path.join(self.run_dir, f"model_solved_{ts}.pth"))
                    np.save(
                        os.path.join(self.run_dir, f"rewards_solved_{ts}.npy"),
                        np.array(self.reward_history),
                    )

                    final_saved = True
                    break

                # -------- MAX EPISODES CONDITION --------
                if episode >= max_episodes:
                    print("Max episodes reached. Saving final model.")

                    ts = int(time.time())
                    self.save_model(os.path.join(self.run_dir, f"model_final_{ts}.pth"))
                    np.save(
                        os.path.join(self.run_dir, f"rewards_final_{ts}.npy"),
                        np.array(self.reward_history),
                    )

                    final_saved = True
                    break

                episode_reward = 0
                state, _ = self.env.reset()
                state = np.asarray(state, dtype=np.float32)

            else:
                state = next_state

            if time.time() - last_save_time >= save_interval:

                # Save model
                ts = int(time.time())
                self.save_model(os.path.join(self.run_dir, f"model_{ts}.pth"))

                 # Save rewards
                np.save(
                    os.path.join(self.run_dir, f"rewards_{ts}.npy"),
                    np.array(self.reward_history)
                )

                last_save_time = time.time()



            if live_plot_enabled and frame % 5000 == 0:
                self.plot_rewards_live()

    # =========================
    # Learning
    # =========================
    def train_step(self):
        batch = self.memory.sample_batch()

        states = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["acts"], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch["rews"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()


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
