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

    

    ckpt_path = "models/CartPole/DDQN/model_solved_1766145727.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    rewards = checkpoint["reward_history"]

    print(rewards)


    # agent = RainbowDQNAgent(config)

    # agent.train(max_frames=1_000_000, verbose=True, save_interval=0.2)







if __name__ == "__main__":
    main()
