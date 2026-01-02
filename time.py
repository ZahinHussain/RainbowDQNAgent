import numpy as np
import matplotlib.pyplot as plt

# ---- Load data ----
timestamps_sec = np.load(
    "runs/run-ALE/Frostbite-v5-20251225-024948/time_stamps_1766685000.npy"
)


# Convert seconds → minutes
timestamps_min = timestamps_sec / 60.0

# Epochs (index-based)
epochs = np.arange(1, len(rewards) + 1)

fig, ax1 = plt.subplots(figsize=(12, 6))

# --- Bottom X: Epochs ---
ax1.plot(epochs, rewards, label="Reward", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Reward")
ax1.grid(True)

# --- Top X: Time (minutes) ---
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())

step = max(1, len(epochs) // 6)
ax2.set_xticks(epochs[::step])
ax2.set_xticklabels([f"{t:.1f} min" for t in timestamps_min[::step]])
ax2.set_xlabel("Time (minutes)")

plt.title("Frostbite Reward over Epochs and Time")
plt.tight_layout()
plt.show()
