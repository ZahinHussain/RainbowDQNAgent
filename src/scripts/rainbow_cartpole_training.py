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
    seed_everything(77)

    config = {
        "env_name": "CartPole-v1",
        "network": "mlp",          

        "batch_size": 64,
        "gamma": 0.99,
        "lr": 5e-4,                
        "tau": 0.01,               

        "memory_size": 5_000,      
        "n_step": 3,               

        "v_min": 0.0,
        "v_max": 500.0,            
        "atom_size": 51,           

        "target_update_type": "hard",
        "target_update_freq": 200,  
    }



    agent = RainbowDQNAgent(config)

    agent.train(max_frames=1_000_000, verbose=True, save_interval=0.2)


if __name__ == "__main__":
    main()
