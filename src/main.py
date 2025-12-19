import gymnasium as gym
from agents.ddqn import DDQNAgent

def main():
    
    config = {
        # Environment
        "env_name": "LunarLander-v3",

        # Optimization
        "batch_size": 64,
        "gamma": 0.99,
        "lr": 1e-3,
        "tau": 0.005,

        # Exploration (ε-greedy)
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 200_000,

        # Replay (PER + n-step)
        "memory_size": 100_000,
        "n_step": 3,

        # Target network updates
        "target_update_type": "hard",   # "hard" or "soft"
        "target_update_freq": 1_000
    }

    agent = DDQNAgent(config)

    agent.train(max_frames=1_000_000, verbose=True)


if __name__ == "__main__":
    main()
