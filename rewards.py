import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(rewards, window=100):
    """
    Plot episode rewards and moving average.

    Args:
        rewards (array-like): episode rewards
        window (int): moving average window
    """
    rewards = np.asarray(rewards)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward", alpha=0.4)

    if len(rewards) >= window:
        moving_avg = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )
        plt.plot(
            range(window - 1, len(rewards)),
            moving_avg,
            label=f"Moving Avg ({window})",
            linewidth=2,
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
