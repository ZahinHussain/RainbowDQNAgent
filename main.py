from rainbowdqn_agent import RainbowDQNAgent
import numpy as np
from rewards import plot_rewards, plot_reward_distribution



def main():

    config = {
        "batch_size": 128,
        "gamma": 0.99,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 200_000,
        "tau": 0.005,
        "lr": 5e-4,
        "num_episodes": 1000,
   
        "env_name": "ALE/Frostbite-v5",     # LunarLander-v3     ALE/Pong-v5     CartPole-v1
    }

 
    # agent = RainbowDQNAgent(config)
    agent = RainbowDQNAgent(config)
    
    # agent.load_model("runs/run-lunar-20251218-105055/model_1766055116.pth")
    # Train
    agent.train(run_name="ALE/Frostbite-v5", max_frames=10_000_000, save_interval = 30, live_plot_enabled=False, verbose=True)

    # # Plot rewards after training
    # rewards = np.load("rewards_1765411435.npy")
    # plot_rewards(rewards)

    # Reward distribution (histogram)
    plot_reward_distribution(
        agent.reward_history,
        bins=60,
        last_n=500,
        xlim=(-600, 400),
        title="Rainbow (ALE/Frostbite-v5): Reward Distribution (last 500 episodes)",
        density=True,
    )

    # # Load the model into a new agent for evaluation
    # eval_agent = DQNAgent()
    # eval_agent.load_model("Outputs/dqn_model.pth")
    
    # # Evaluate or run the agent
    # eval_agent.evaluate()

if __name__ == "__main__":
    main()
