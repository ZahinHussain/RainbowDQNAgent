




from rainbowdqn_agent import RainbowDQNAgent
import numpy as np


def main():

    config = {
        "batch_size": 32,
        "gamma": 0.99,
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 5000,
        "tau": 0.005,
        "lr": 1e-3,
        "num_episodes": 500,
        "env_name": "ALE/Pong-v5"
    }

    agent = RainbowDQNAgent("ALE/Pong-v5", config)

    # Train
    agent.train(max_frames=1400, live_plot_enabled=False, verbose=True)

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