import numpy as np


class ExplorationOnlyAgent:
    """
    An agent that follows an exploration-only strategy for the Multi-Armed Bandit problem.
    In this strategy, the agent randomly selects actions with equal probability,
    without considering the past outcomes or estimated rewards.
    """

    def __init__(
        self,
        num_actions: int,
        num_trials: int,
        action_probabilities: list,
        reward_value: float = 1.0,
    ):
        """
        Initialize the exploration-only agent with the given parameters.

        Args:
            num_actions (int): Total number of available actions (bandit arms).
            num_trials (int): Number of trials (attempts) per episode.
            action_probabilities (list): A list of success probabilities for each action.
            reward_value (float): Reward obtained on a successful action (default is 1.0).
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
        Execute one episode where the agent randomly chooses an action at each trial.

        Returns:
            float: Total accumulated reward from the episode.
        """
        episode_return = 0.0

        # Perform the specified number of trials in this episode.
        for _ in range(self.num_trials):
            # Randomly pick an action index between 0 and (num_actions - 1)
            action = np.random.randint(0, self.num_actions)

            # Generate a random float in [0, 1). If this value is less than the
            # selected action's success probability, the action is successful.
            if np.random.random() < self.action_probabilities[action]:
                # Increment the episode's reward by the reward value.
                episode_return += self.reward_value

        return episode_return

    def calculate_regret(self, actual_return: float) -> float:
        """
        Compute the regret, which is the difference between the optimal return and the actual return.

        Args:
            actual_return (float): Total reward obtained by the agent.

        Returns:
            float: Regret value representing the performance gap to the optimal strategy.
        """
        return self.optimal_return - actual_return

    def run_multiple_episodes(self, num_episodes: int) -> tuple[float, float]:
        """
        Execute multiple episodes and compute the average return and regret.

        Args:
            num_episodes (int): The number of episodes to run.

        Returns:
            tuple[float, float]: A tuple containing:
                - average_return: Mean reward across episodes.
                - average_regret: Regret calculated based on the average return.
        """
        total_return = 0.0

        # Run each episode and accumulate the total return.
        for _ in range(num_episodes):
            episode_return = self.run_episode()
            total_return += episode_return

        # Calculate the average return over all episodes.
        average_return = total_return / num_episodes

        # Determine the average regret based on the average return.
        average_regret = self.calculate_regret(average_return)

        return average_return, average_regret

    def calculate_theoretical_values(self) -> tuple[float, float]:
        """
        Calculate the theoretical expected return and corresponding regret if each action
        were selected exactly 25% of the time (assuming 4 actions).

        Returns:
            tuple[float, float]: A tuple containing:
                - theoretical_return: The sum of expected rewards from each action.
                - theoretical_regret: Regret computed from the theoretical return.
        """
        # For each action, assume it is chosen exactly 25 times in 100 trials (25% selection rate)
        equal_selection = self.num_trials / self.num_actions
        theoretical_return = sum(
            equal_selection * p * self.reward_value for p in self.action_probabilities
        )

        theoretical_regret = self.calculate_regret(theoretical_return)

        return theoretical_return, theoretical_regret


if __name__ == "__main__":
    # Set up the agent with 4 actions, 100 trials per episode,
    # and the specified success probabilities for each action.
    agent = ExplorationOnlyAgent(
        num_actions=4,
        num_trials=100,
        action_probabilities=[0.5, 0.7, 0.3, 0.4],
        reward_value=1.0,
    )

    # Run a single episode and calculate its return and regret.
    single_return = agent.run_episode()
    single_regret = agent.calculate_regret(single_return)
    print(f"Single Episode - Return: {single_return:.2f}, Regret: {single_regret:.2f}")

    # Calculate and display the theoretical return and regret values.
    theo_return, theo_regret = agent.calculate_theoretical_values()

    # Run multiple episodes (e.g., 1000) and compute the average performance metrics.
    num_episodes = 1000
    avg_return, avg_regret = agent.run_multiple_episodes(num_episodes=num_episodes)
    print(f"\nAverage over {num_episodes} episodes:")
    print(f"Expected Return: {avg_return:.2f} (theoretical: {theo_return:.2f})")
    print(f"Average Regret: {avg_regret:.2f} (theoretical: {theo_regret:.2f})")
