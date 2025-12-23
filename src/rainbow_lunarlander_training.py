from agents.ddqn import DDQNAgent
from agents.rainbow import RainbowDQNAgent

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
    seed = 2
    seed_everything(seed)

    config = {
        "env_name": "LunarLander-v3",
        "network": "mlp",          

        "batch_size": 32,
        "gamma": 0.99,
        "lr": 1e-3,                
        "tau": 0.05,               

        "memory_size": 100000,      
        "n_step": 3,               

        "v_min": -300,
        "v_max": 300.0,            
        "atom_size": 31,           

        "target_update_type": "hard",
        "target_update_freq": 1000,  
    }



    agent = RainbowDQNAgent(config, seed=seed)

    agent.train(max_episodes=1000, verbose=True, save_interval=5)


if __name__ == "__main__":
    main()
