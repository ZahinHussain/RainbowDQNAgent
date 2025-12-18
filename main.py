from rainbowdqn_agent import RainbowDQNAgent
import numpy as np
from rewards import plot_rewards, plot_reward_distribution



def main():

    config = {
        "batch_size": 32,
        "gamma": 0.99,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 2000,
        "tau": 0.005,
        "lr": 5e-4,
        "num_episodes": 1000,
        "target_update_type": "hard",   # "hard" or "soft"
        "target_update_freq": 5000,     # steps (only for hard)
        "env_name": "LunarLander-v3",     # LunarLander-v3     ALE/Pong-v5     CartPole-v1
    }

 
    # agent = RainbowDQNAgent(config)
    agent = RainbowDQNAgent(config)
    
    # agent.load_model("runs/run-lunar-20251218-105055/model_1766055116.pth")
    # Train
    agent.train(run_name="lunar", max_frames=10000000000, save_interval = 5, live_plot_enabled=True, verbose=True)

    # # Plot rewards after training
    # rewards = np.load("rewards_1765411435.npy")
    # plot_rewards(rewards)

    # Reward distribution (histogram)
    plot_reward_distribution(
        agent.reward_history,
        bins=60,
        last_n=500,
        xlim=(-600, 400),
        title="Rainbow (LunarLander): Reward Distribution (last 500 episodes)",
        density=True,
    )

    # # Load the model into a new agent for evaluation
    # eval_agent = DQNAgent()
    # eval_agent.load_model("Outputs/dqn_model.pth")
    
    # # Evaluate or run the agent
    # eval_agent.evaluate()

if __name__ == "__main__":
    main()
