import gymnasium as gym
from agents.ddqn import DDQNAgent

import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic (important)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: enforce deterministic algorithms (PyTorch ≥1.8)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    # Hash seed (affects Python internals)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    seed = 0
    seed_everything(0)

    config = {
        "env_name": "LunarLander-v3",

        "batch_size": 32,
        "gamma": 0.99,
        "lr": 1e-3,

        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 2000,  # steps

        "memory_size": 10_000,
        "n_step": 1,

        "target_update_type": "hard",
        "target_update_freq": 100,
    }


    agent = DDQNAgent(config, seed=seed)

    agent.train(max_frames=1_000_000, save_interval=0.2 ,verbose=True)


if __name__ == "__main__":
    main()
