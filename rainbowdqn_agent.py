import gymnasium as gym
from  gymnasium.wrappers import AtariPreprocessing ,FrameStackObservation
import torch
import numpy as np
from distributed import C51Distribution
from network import C51NoisyDuelingCNN, C51NoisyDuelingMLP
from datetime import datetime
import ale_py
from replay_buffer import NStepPrioritizedReplayBuffer
import time
import os
from datetime import datetime

import torch.optim as optim
from matplotlib import pyplot as plt



class RainbowDQNAgent:
    def __init__(self, config):


        self.batch_size   = config["batch_size"]
        self.gamma        = config["gamma"]
        self.tau          = config["tau"]
        self.lr           = config["lr"]
        self.env_id       = config["env_name"]

        self.target_update_type = config.get("target_update_type", "hard")
        self.target_update_freq = config.get("target_update_freq", 5000)
        self.learn_steps = 0


        self.is_atari = self.env_id.startswith("ALE/")

        if self.is_atari:
            base_env = gym.make(self.env_id, frameskip=1)

            env = AtariPreprocessing(
                base_env,
                grayscale_obs=True,
                scale_obs=True,
                frame_skip=4,
                screen_size=84
            )

            env = FrameStackObservation(env, stack_size=4)

        else:
            env = gym.make(self.env_id)

        self.env = env

        # Observation shape = (4, 84, 84)
        self.obs_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
        self.c51 = C51Distribution(
           v_min=-10,
            v_max=10,
            atom_size=51,
            device=self.device
        )

        self.atom_size = self.c51.atom_size


        if self.is_atari:
            self.policy_net = C51NoisyDuelingCNN(
                num_actions=self.num_actions,
                atom_size=self.atom_size,
                support=self.c51.support
            ).to(self.device)

            self.target_net = C51NoisyDuelingCNN(
                num_actions=self.num_actions,
                atom_size=self.atom_size,
                support=self.c51.support
            ).to(self.device)

        else:
            obs_dim = self.obs_shape[0]

            self.policy_net = C51NoisyDuelingMLP(
                obs_dim=obs_dim,
                num_actions=self.num_actions,
                atom_size=self.atom_size,
                support=self.c51.support
            ).to(self.device)

            self.target_net = C51NoisyDuelingMLP(
                obs_dim=obs_dim,
                num_actions=self.num_actions,
                atom_size=self.atom_size,
                support=self.c51.support
            ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.memory = NStepPrioritizedReplayBuffer(
            capacity=375000,
            obs_shape=self.obs_shape,
            n_step=10,
            gamma=self.gamma
        )

        self.steps_done = 0
        self.reward_history = []

        self.run_dir = None
        self.run_name = None


    def select_action(self, state):
        # NoisyNet exploration
        self.policy_net.reset_noise()

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist = self.policy_net(state_t)
            q_vals = self.c51.expectation(dist)
            return q_vals.argmax(1).item()

    def train(self, max_frames=5_000_000, save_interval = 30, live_plot_enabled=False, verbose=False, run_name="default"):

        self.run_name = run_name

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join("runs", f"run-{run_name}-{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        if verbose:
            print(f"[RUN] Saving to {self.run_dir}")

        last_save_time = time.time()
        save_interval = save_interval * 60

        frame = 0
        episode_reward = 0

        state, _ = self.env.reset()
        state = np.array(state, dtype=np.float32, copy=False)

        while frame < max_frames:

            action = self.select_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            reward = np.clip(reward, -1.0, 1.0)
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            done = terminated or truncated

            frame += 1
            episode_reward += reward

            # Store transition
            self.memory.store(state, action, reward, next_state, done)

            # Learn
            if len(self.memory) >= self.batch_size:
                self.train_step()
                
                if self.target_update_type == "soft":
                    self.soft_update()
                    self.learn_steps += 1

                elif self.target_update_type == "hard":
                    if self.learn_steps % self.target_update_freq == 0:
                        self.hard_update()

            # Continue or reset episode
            if not done:
                state = next_state
                ep = len(self.reward_history)
                avg_100 = np.mean(self.reward_history[-100:])
            else:
                self.reward_history.append(episode_reward)

                if verbose:
                    ep = len(self.reward_history)
                    avg_100 = np.mean(self.reward_history[-100:])


                    
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(f"[{ts}] Episode {ep} | Reward: {episode_reward:.2f}")

                episode_reward = 0
                state, _ = self.env.reset()
                state = np.array(state, dtype=np.float32, copy=False)


       

            # Auto-save every 30 minutes
            if time.time() - last_save_time >= save_interval:
                timestamp = int(time.time())

                self.save_model(os.path.join(self.run_dir, f"model_{timestamp}.pth"))

                np.save(
                    os.path.join(self.run_dir, f"rewards_{timestamp}.npy"),
                    np.array(self.reward_history)
                )

  
                if verbose:
                    print(f"[AUTO-SAVE] Model and rewards saved at {timestamp}")

                last_save_time = time.time()

            # Live plot
            if live_plot_enabled and frame % 5000 == 0:
                self.plot_rewards_live()

    def train_step(self):

        (states, actions, rewards, next_states, dones, gammas, idxs, weights,) = self.memory.sample(self.batch_size)

        states      = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)
        gammas      = torch.tensor(gammas, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights     = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Current dist
        dist = self.policy_net(states)
        dist_a = dist[range(states.size(0)), actions.squeeze()]

        # Next action chosen by policy net (double DQN)
        with torch.no_grad():
            next_dist_policy = self.policy_net(next_states)
            next_q = self.c51.expectation(next_dist_policy)
            next_actions = next_q.argmax(dim=1)

            next_dist_target = self.target_net(next_states)
            next_dist_a = next_dist_target[range(states.size(0)), next_actions]

            # C51 projection
            target_dist = self.c51.project(next_dist_a, rewards, dones, gammas)

        # Loss
        loss_elem = -(target_dist * (dist_a + 1e-8).log()).sum(dim=1)
        loss = (loss_elem * weights.squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update PER priorities
        td = loss_elem.detach().cpu().numpy()
        self.memory.update_priorities(idxs, td)

    def hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self):
        for name, param in self.policy_net.state_dict().items():
            t = self.target_net.state_dict()[name]
            self.target_net.state_dict()[name].copy_(
                param * self.tau + t * (1 - self.tau)
            )

    def plot_rewards_live(self):
        plt.clf()
        plt.title("Training Rewards")
        plt.plot(self.reward_history)
        plt.pause(0.01)

    def save_model(self, path):
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "frame": getattr(self, "frame", None),
            "reward_history": getattr(self, "reward_history", None),
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "reward_history" in checkpoint and checkpoint["reward_history"] is not None:
            self.reward_history = checkpoint["reward_history"]

        print(f"Loaded model from {path}")

    def play(self, episodes=5, sleep=0.01, render=True):
        # Create render env
        base_env = gym.make(
            self.env.unwrapped.spec.id,
            frameskip=1,
            render_mode="human" if render else None
        )

        env = AtariPreprocessing(
            base_env,
            grayscale_obs=True,
            scale_obs=True,
            frame_skip=4,
            screen_size=84
        )
        env = FrameStackObservation(env, num_stack=4)

        self.policy_net.eval()
        self.target_net.eval()

        for ep in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            done = False
            ep_reward = 0

            while not done:
                with torch.no_grad():
                    state_t = torch.tensor(
                        state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    dist = self.policy_net(state_t)
                    q_vals = self.c51.expectation(dist)
                    action = q_vals.argmax(1).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = np.array(next_state, dtype=np.float32)
                ep_reward += reward

                if render:
                    time.sleep(sleep)

            print(f"[PLAY] Episode {ep+1} | Reward: {ep_reward:.2f}")

        env.close()

    def evaluate(self, episodes=20, render=False, sleep=0.0):
        """
        Run evaluation episodes and return episode rewards.
        """

        base_env = gym.make(
            self.env.unwrapped.spec.id,
            frameskip=1,
            render_mode="human" if render else None
        )

        env = AtariPreprocessing(
            base_env,
            grayscale_obs=True,
            scale_obs=True,
            frame_skip=4,
            screen_size=84
        )
        env = FrameStackObservation(env, num_stack=4)

        self.policy_net.eval()
        self.target_net.eval()

        episode_rewards = []

        for ep in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            done = False
            ep_reward = 0.0

            while not done:
                with torch.no_grad():
                    state_t = torch.tensor(
                        state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    # deterministic policy (NO reset_noise)
                    dist = self.policy_net(state_t)
                    q_vals = self.c51.expectation(dist)
                    action = q_vals.argmax(1).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = np.array(next_state, dtype=np.float32)
                ep_reward += reward

                if render:
                    time.sleep(sleep)

            episode_rewards.append(ep_reward)

        env.close()
        return episode_rewards
