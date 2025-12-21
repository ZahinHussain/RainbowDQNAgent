import numpy as np
import matplotlib.pyplot as plt

# Load rewards
rewards = np.load("runs/run-ALE/Frostbite-v5-20251219-185038/rewards_1766186440.npy")

# Plot rewards
plt.figure(figsize=(12,6))
plt.plot(rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Frostbite Rewards over Episodes")
plt.legend()
plt.grid(True)
plt.show()
