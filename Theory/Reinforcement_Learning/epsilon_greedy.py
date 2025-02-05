import numpy as np


class EpsilonGreedyAgent:
    """
    An agent that implements the epsilon-greedy strategy for the Multi-Armed Bandit problem.

    This strategy balances exploration and exploitation by:
      - With probability ε: exploring (choosing a random action)
      - With probability 1-ε: exploiting (choosing the action with the highest estimated value)

    The agent tracks:
      - Q-values (Q_a): Estimated value for each action.
      - Action counts (N_a): How many times each action has been selected.
      - Reward history: A matrix that records rewards for each action at each trial.
    """

    def __init__(
        self,
        num_actions: int,
        num_trials: int,
        action_probabilities: list,
        epsilon: float = 0.1,
        reward_value: float = 1.0,
    ):
        """
        Initialize the epsilon-greedy agent with given parameters.

        Args:
            num_actions (int): Total number of available actions (bandit arms).
            num_trials (int): Total number of trials per episode.
            action_probabilities (list): Success probability for each action.
            epsilon (float): Probability of choosing a random action (exploration).
            reward_value (float): Reward received when an action is successful.
        """
        self.num_actions = num_actions
        self.num_trials = num_trials
        self.action_probabilities = action_probabilities
        self.epsilon = epsilon
        self.reward_value = reward_value

        # Calculate the maximum achievable reward (optimal strategy)
        # by always choosing the action with the highest probability.
        self.optimal_return = (
            max(self.action_probabilities) * self.reward_value * self.num_trials
        )

    def run_episode(self) -> float:
        """
        Execute a single episode using the epsilon-greedy strategy.

        Returns:
            float: The total reward accumulated during the episode.
        """
        # Initialize estimated Q-values for each action (starting at 0)
        Q_a = np.zeros(self.num_actions)
        # Initialize the count for each action (how many times each action has been chosen)
        N_a = np.zeros(self.num_actions)
        # Initialize a matrix to store rewards per action for each trial (for record keeping)
        reward_matrix = np.zeros((self.num_actions, self.num_trials))
        # Initialize the total reward for the episode
        episode_return = 0.0

        # Run through each trial in the episode
        for k in range(self.num_trials):
            # Decide whether to explore or exploit:
            # - With probability epsilon, choose a random action (exploration)
            # - Otherwise, choose the action with the highest estimated Q-value (exploitation)
            if np.random.random() <= self.epsilon:
                action = np.random.randint(0, self.num_actions)
            else:
                # Identify the maximum Q-value among all actions.
                max_value = np.max(Q_a)
                # Get indices of actions that share the maximum value.
                max_indices = np.where(Q_a == max_value)[0]
                # Randomly choose one among them to break any ties.
                action = np.random.choice(max_indices)

            # Record that this action has been chosen by incrementing its count.
            N_a[action] += 1

            # Simulate the trial: determine if the chosen action is successful
            if np.random.random() < self.action_probabilities[action]:
                # On success, record the reward for this action at the current trial.
                reward_matrix[action, k] = self.reward_value
                # Add the reward to the episode's total return.
                episode_return += self.reward_value

                # Update the estimated Q-value for the chosen action.
                # If this is the first trial for the action, simply use the current reward.
                # Otherwise, average the sum of rewards obtained so far with the current reward.
                if k == 0:
                    Q_a[action] = self.reward_value / N_a[action]
                else:
                    previous_rewards = np.sum(reward_matrix[action, :k])
                    Q_a[action] = (self.reward_value + previous_rewards) / N_a[action]
            else:
                # On failure (no reward), update the Q-value using only past rewards.
                # For the first trial (k == 0), Q_a remains 0 (as no reward was obtained).
                if k > 0:
                    previous_rewards = np.sum(reward_matrix[action, :k])
                    Q_a[action] = previous_rewards / N_a[action]

        return episode_return

    def calculate_regret(self, actual_return: float) -> float:
        """
        Calculate the regret compared to the optimal strategy.

        Regret is defined as the difference between the return of the optimal strategy
        (always choosing the best action) and the actual return achieved.

        Args:
            actual_return (float): The total reward achieved in the episode.

        Returns:
            float: The calculated regret.
        """
        return self.optimal_return - actual_return

    def run_multiple_episodes(self, num_episodes: int) -> tuple[float, float]:
        """
        Execute multiple episodes and compute the average return and average regret.

        Args:
            num_episodes (int): The number of episodes to simulate.

        Returns:
            tuple[float, float]: A tuple containing:
                - average_return: Mean reward over all episodes.
                - average_regret: Regret computed from the average return.
        """
        total_return = 0.0

        # Simulate the specified number of episodes
        for _ in range(num_episodes):
            episode_return = self.run_episode()
            total_return += episode_return

        # Calculate the average return per episode.
        average_return = total_return / num_episodes
        # Calculate the corresponding regret.
        average_regret = self.calculate_regret(average_return)

        return average_return, average_regret


if __name__ == "__main__":
    # Initialize the epsilon-greedy agent with:
    # - 4 actions (bandit arms)
    # - 100 trials per episode
    # - Given success probabilities for each action
    # - Exploration rate (epsilon) of 0.1
    # - Reward of 1.0 for successful actions
    agent = EpsilonGreedyAgent(
        num_actions=4,
        num_trials=100,
        action_probabilities=[0.5, 0.7, 0.3, 0.4],
        epsilon=0.1,
        reward_value=1.0,
    )

    # Run a single episode and output its return and computed regret.
    single_return = agent.run_episode()
    single_regret = agent.calculate_regret(single_return)
    print(f"Single Episode - Return: {single_return:.2f}, Regret: {single_regret:.2f}")

    # Run multiple episodes (e.g., 10,000) and compute the average return and regret.
    num_episodes = 10000
    avg_return, avg_regret = agent.run_multiple_episodes(num_episodes=num_episodes)
    print(f"\nAverage over {num_episodes} episodes:")
    print(f"Expected Return: {avg_return:.2f}")
    print(f"Average Regret: {avg_regret:.2f}")
