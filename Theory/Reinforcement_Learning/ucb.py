import numpy as np

"""
UCB Agent for the Multi-Armed Bandit Problem
--------------------------------------------

This module implements an agent that uses the UCB (Upper Confidence Bound) strategy 
to solve the multi-armed bandit problem. In this context, an agent must choose among 
several actions (or "arms") over a series of trials, balancing the trade-off between 
exploration (gathering information about each arm) and exploitation (selecting the best 
known arm).

The UCB strategy selects the action 'a' that maximizes:
    Q(a) + c * sqrt(ln(t) / N(a))
where:
    - Q(a) is the current estimated value (mean reward) of action a.
    - t is the current trial number (or time step).
    - N(a) is the number of times action a has been chosen.
    - c is an exploration parameter (commonly set to sqrt(2)), which scales the 
      exploration bonus.

The exploration bonus term, sqrt(ln(t) / N(a)), is derived from Hoeffding's inequality,
which provides a bound on the probability that the estimated mean deviates from the true 
mean. This term ensures that actions that have been tried less frequently (small N(a)) have 
a higher chance of being selected, but as they are tried more often, the bonus shrinks, and 
the agent increasingly favors actions with a higher observed reward.

In contrast, the epsilon-greedy strategy explores randomly with a fixed probability (ε) 
by selecting any action at random, even if that action has been proven to perform poorly.
This means that even when some actions consistently yield low rewards, epsilon-greedy 
may continue to explore them purely by chance. Such inefficiency can be particularly 
detrimental when the number of trials is large, as the agent might waste valuable opportunities 
exploring bad actions instead of focusing on the best ones.

By leveraging the UCB algorithm, the agent avoids the pitfall of repeatedly exploring the 
same suboptimal actions. Instead, it uses the statistical confidence (based on Hoeffding's inequality)
to dynamically balance exploration and exploitation. As a result, UCB is typically more 
efficient than epsilon-greedy, especially in environments where the cost of exploring poor 
actions is high.

The implementation below initializes each action at least once (to avoid division-by-zero in 
the UCB computation) and then proceeds to use the UCB criterion to choose actions over subsequent 
trials, updating estimates and counts along the way.
"""


class UCBAgent:
    """
    An agent that implements the UCB (Upper Confidence Bound) strategy for the Multi-Armed Bandit problem.

    UCB balances exploration and exploitation by selecting actions that maximize:
        Q(a) + c * sqrt(ln(t)/N(a))
    where:
        - Q(a): Estimated value (mean reward) of action a.
        - t: Current trial number.
        - N(a): Number of times action a has been chosen.
        - c: Exploration parameter (here, c is implicitly set by sqrt(2) in the formula).

    The agent maintains:
      - Q-values (Q_a): The estimated value for each action.
      - Action counts (N_a): The number of times each action was selected.
      - Upper bounds (U_a): The exploration bonus for each action.
      - Reward history: A matrix recording the rewards obtained for each action at each trial.
    """

    def __init__(
        self,
        num_actions: int,
        num_trials: int,
        action_probabilities: list,
        reward_value: float = 1.0,
    ):
        """
        Initialize the UCB agent with the specified parameters.

        Args:
            num_actions (int): Total number of available actions (bandit arms).
            num_trials (int): Total number of trials per episode.
            action_probabilities (list): Success probability for each action.
            reward_value (float): Reward received for a successful action (default is 1.0).
        """
        self.num_actions = num_actions
        self.num_trials = num_trials
        self.action_probabilities = action_probabilities
        self.reward_value = reward_value

        # Calculate the maximum achievable reward (optimal strategy)
        # by always choosing the action with the highest probability.
        self.optimal_return = (
            max(self.action_probabilities) * self.reward_value * self.num_trials
        )

    def run_episode(self) -> float:
        """
        Execute one episode using the UCB strategy.

        Returns:
            float: The total reward accumulated over the episode.
        """
        # Initialize Q-values (estimated mean reward) for each action to zero.
        Q_a = np.zeros(self.num_actions)
        # Initialize the action counts for each action to zero.
        N_a = np.zeros(self.num_actions)
        # Create a matrix to record the reward received for each action at each trial.
        reward_matrix = np.zeros((self.num_actions, self.num_trials))
        # Initialize the total reward for the episode.
        episode_return = 0.0

        # --- Initialization Phase ---
        # To avoid division-by-zero in the UCB calculation,
        # we try each action once during the first 'num_actions' trials.
        for action in range(self.num_actions):
            # For trial index equal to the current action index,
            # simulate the outcome for this action.
            if np.random.random() < self.action_probabilities[action]:
                # If the action is successful, record the reward.
                reward_matrix[action, action] = self.reward_value
                # Increase the total reward.
                episode_return += self.reward_value
                # Set the Q-value to the reward received (since N_a will be 1).
                Q_a[action] = self.reward_value
            # Mark that this action was tried once.
            N_a[action] = 1

        # --- Main Loop ---
        # Now that every action has been tried once, run the remaining trials.
        for k in range(self.num_actions, self.num_trials):
            # Compute the UCB value for each action.
            # UCB_value = Q_a + sqrt(2 * ln(t) / N_a), where t = k+1 (to avoid ln(0))
            UCB_values = Q_a + np.sqrt(2 * np.log(k + 1) / N_a)

            # Select the action with the highest UCB value.
            max_value = np.max(UCB_values)
            max_indices = np.where(UCB_values == max_value)[0]
            # If multiple actions have the same UCB value, choose one at random.
            action = np.random.choice(max_indices)

            # Increment the count for the selected action.
            N_a[action] += 1

            # Simulate the outcome for the chosen action.
            if np.random.random() < self.action_probabilities[action]:
                # If the action is successful, record the reward.
                reward_matrix[action, k] = self.reward_value
                episode_return += self.reward_value

                # Update Q-value by averaging all rewards obtained for this action so far.
                previous_rewards = np.sum(reward_matrix[action, :k])
                Q_a[action] = (previous_rewards + self.reward_value) / N_a[action]
            else:
                # On failure (no reward), update the Q-value using only past rewards.
                previous_rewards = np.sum(reward_matrix[action, :k])
                Q_a[action] = previous_rewards / N_a[action]

            # No need to update U_a explicitly here since it's computed from Q_a and N_a in the next iteration.

        return episode_return

    def calculate_regret(self, actual_return: float) -> float:
        """
        Calculate the regret relative to the optimal strategy.

        Args:
            actual_return (float): The total reward obtained in the episode.

        Returns:
            float: The regret, defined as the difference between the optimal return and the actual return.
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

        # Run the specified number of episodes.
        for _ in range(num_episodes):
            episode_return = self.run_episode()
            total_return += episode_return

        # Calculate the average return.
        average_return = total_return / num_episodes
        # Calculate the average regret.
        average_regret = self.calculate_regret(average_return)

        return average_return, average_regret


if __name__ == "__main__":
    # Initialize the UCB agent with:
    # - 4 actions (bandit arms)
    # - 100 trials per episode
    # - Given success probabilities for each action
    # - A reward value of 1.0 for each successful trial
    agent = UCBAgent(
        num_actions=4,
        num_trials=100,
        action_probabilities=[0.5, 0.7, 0.3, 0.4],
        reward_value=1.0,
    )

    # Run a single episode and print its return and regret.
    single_return = agent.run_episode()
    single_regret = agent.calculate_regret(single_return)
    print(f"Single Episode - Return: {single_return:.2f}, Regret: {single_regret:.2f}")

    # Run multiple episodes (e.g., 10,000) and print average return and regret.
    num_episodes = 10000
    avg_return, avg_regret = agent.run_multiple_episodes(num_episodes=num_episodes)
    print(f"\nAverage over {num_episodes} episodes:")
    print(f"Expected Return: {avg_return:.2f}")
    print(f"Average Regret: {avg_regret:.2f}")
