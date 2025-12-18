from rainbowdqn_agent import RainbowDQNAgent
import numpy as np


def main():

    config = {
        "batch_size": 32,
        "gamma": 0.99,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 2000,
        "tau": 0.005,
        "lr": 1e-3,
        "num_episodes": 1000,
        "env_name": "CartPole-v1",     # Lunar.   ALE/Pong-v5
    }


    # agent = RainbowDQNAgent(config)
    agent = RainbowDQNAgent(config)
    
    # agent.load_model("runs/run-lunar-20251218-105055/model_1766055116.pth")
    # Train
    agent.train(run_name="lunar", max_frames=1000000, save_interval = 5, live_plot_enabled=True, verbose=True)

    # # Plot rewards after training
    # rewards = np.load("Outputs/09-12-25_00-45-36/reward_history100.npy")
    # plot_rewards(rewards)

    # # Load the model into a new agent for evaluation
    # eval_agent = DQNAgent()
    # eval_agent.load_model("Outputs/dqn_model.pth")
    
    # # Evaluate or run the agent
    # eval_agent.evaluate()

if __name__ == "__main__":
    main()