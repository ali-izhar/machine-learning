import numpy as np
import matplotlib.pyplot as plt
from exploration_only import ExplorationOnlyAgent
from exploitation_only import ExploitationOnlyAgent
from epsilon_greedy import EpsilonGreedyAgent
from ucb import UCBAgent


def compare_strategies(
    num_episodes: int,
    trial_counts: list,
    action_probabilities: list,
    epsilons: list,
    reward_value: float = 1.0,
):
    """
    Compare different multi-armed bandit strategies across various numbers of trials.

    Args:
        num_episodes (int): Number of episodes to average over
        trial_counts (list): List of different trial counts to test
        action_probabilities (list): Success probability for each action
        epsilons (list): List of epsilon values for epsilon-greedy strategy
        reward_value (float): Reward value for successful actions
    """
    num_actions = len(action_probabilities)
    optimal_prob = max(action_probabilities)

    # Initialize arrays to store performance ratios
    ratios = {
        "exploration": np.zeros(len(trial_counts)),
        "exploitation": np.zeros(len(trial_counts)),
        "ucb": np.zeros(len(trial_counts)),
    }
    # Add entries for each epsilon value
    for eps in epsilons:
        ratios[f"eps_{eps}"] = np.zeros(len(trial_counts))

    # Test each trial count
    for i, num_trials in enumerate(trial_counts):
        optimal_return = optimal_prob * num_trials * reward_value
        returns = {
            "exploration": 0.0,
            "exploitation": 0.0,
            "ucb": 0.0,
        }
        for eps in epsilons:
            returns[f"eps_{eps}"] = 0.0

        # Run episodes
        for _ in range(num_episodes):
            # Exploration only
            agent = ExplorationOnlyAgent(
                num_actions=num_actions,
                num_trials=num_trials,
                action_probabilities=action_probabilities,
                reward_value=reward_value,
            )
            returns["exploration"] += agent.run_episode()

            # Exploitation only
            agent = ExploitationOnlyAgent(
                num_actions=num_actions,
                num_trials=num_trials,
                action_probabilities=action_probabilities,
                reward_value=reward_value,
            )
            returns["exploitation"] += agent.run_episode()

            # UCB
            agent = UCBAgent(
                num_actions=num_actions,
                num_trials=num_trials,
                action_probabilities=action_probabilities,
                reward_value=reward_value,
            )
            returns["ucb"] += agent.run_episode()

            # Epsilon-greedy with different epsilon values
            for eps in epsilons:
                agent = EpsilonGreedyAgent(
                    num_actions=num_actions,
                    num_trials=num_trials,
                    action_probabilities=action_probabilities,
                    epsilon=eps,
                    reward_value=reward_value,
                )
                returns[f"eps_{eps}"] += agent.run_episode()

        # Calculate average returns and ratios
        for key in returns:
            avg_return = returns[key] / num_episodes
            ratios[key][i] = avg_return / optimal_return

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot exploration and exploitation
    plt.plot(trial_counts, ratios["exploration"], "--m*", label="exploration only")
    plt.plot(trial_counts, ratios["exploitation"], "-.k^", label="exploitation only")

    # Plot epsilon-greedy variants
    colors = ["b", "g", "r", "c"]
    for eps, color in zip(epsilons, colors):
        plt.plot(
            trial_counts,
            ratios[f"eps_{eps}"],
            f"-{color}+",
            label=f"epsilon-greedy with {eps}",
        )

    # Plot UCB
    plt.plot(trial_counts, ratios["ucb"], "-r<", label="UCB1")

    plt.xscale("log")
    plt.grid(True)
    plt.xlabel("Number of trials")
    plt.ylabel("Ratio to the optimal strategy")
    plt.legend(fontsize=10)
    plt.ylim(0.5, 1.0)
    plt.title("Comparison of Multi-Armed Bandit Strategies")
    plt.show()


if __name__ == "__main__":
    # Problem parameters
    NUM_EPISODES = 1000
    TRIAL_COUNTS = [50, 100, 200, 300, 400, 500, 1000, 10000]
    ACTION_PROBS = [0.5, 0.7, 0.3, 0.4]
    EPSILONS = [0.01, 0.1, 0.2, 1.0]
    REWARD_VALUE = 1.0

    # Run comparison
    compare_strategies(
        num_episodes=NUM_EPISODES,
        trial_counts=TRIAL_COUNTS,
        action_probabilities=ACTION_PROBS,
        epsilons=EPSILONS,
        reward_value=REWARD_VALUE,
    )
