import numpy as np
from enum import Enum


class DistributionType(Enum):
    """Supported probability distributions for Thompson Sampling."""

    BETA = "beta"
    NORMAL = "normal"


class ThompsonSamplingAgent:
    """
    Thompson Sampling agent supporting multiple distribution types:

    1. Beta Distribution:
       - Natural choice for Bernoulli bandits (0/1 rewards)
       - Parameters: α (successes + 1), β (failures + 1)
       - Domain: [0,1], perfect for probabilities

    2. Normal Distribution:
       - Suitable for continuous rewards
       - Parameters: μ (mean), σ² (variance)
       - Domain: (-∞,∞), needs more samples for stable estimates
    """

    def __init__(
        self,
        num_actions: int,
        dist_type: DistributionType = DistributionType.BETA,
        sigma: float = 1.0,  # For normal distribution
    ):
        """
        Initialize agent with chosen distribution type.

        Args:
            num_actions (int): Number of available actions
            dist_type (DistributionType): Type of probability distribution to use
            sigma (float): Standard deviation for normal distribution
        """
        self.num_actions = num_actions
        self.dist_type = dist_type

        if dist_type == DistributionType.BETA:
            # Beta distribution parameters
            self.successes = np.ones(num_actions)  # α parameters
            self.failures = np.ones(num_actions)  # β parameters
        else:  # NORMAL
            # Normal distribution parameters
            self.means = np.zeros(num_actions)  # μ parameters
            self.counts = np.zeros(num_actions)  # Number of samples
            self.sums = np.zeros(num_actions)  # Sum of rewards
            self.sigma = sigma  # Fixed σ parameter

    def select_action(self) -> int:
        """
        Sample from probability distributions and select best action.

        Returns:
            int: Selected action index
        """
        if self.dist_type == DistributionType.BETA:
            # Sample from Beta(α,β) for each action
            samples = np.random.beta(self.successes, self.failures)
        else:  # NORMAL
            # Sample from Normal(μ, σ²/n) for each action
            std_devs = self.sigma / np.sqrt(
                self.counts + 1
            )  # Add 1 to avoid division by zero
            samples = np.random.normal(self.means, std_devs)

        return np.argmax(samples)

    def update(self, action: int, reward: float):
        """
        Update distribution parameters based on observed reward.

        Args:
            action (int): The action that was taken
            reward (float): The reward received
        """
        if self.dist_type == DistributionType.BETA:
            # Binary reward update for Beta
            if reward > 0:
                self.successes[action] += 1  # Increment α
            else:
                self.failures[action] += 1  # Increment β
        else:  # NORMAL
            # Continuous reward update for Normal
            self.counts[action] += 1
            self.sums[action] += reward
            self.means[action] = self.sums[action] / self.counts[action]

    def get_best_action(self) -> int:
        """
        Get current best action based on distribution means.

        Returns:
            int: Action with highest expected value
        """
        if self.dist_type == DistributionType.BETA:
            expected_values = self.successes / (self.successes + self.failures)
        else:  # NORMAL
            expected_values = self.means

        return np.argmax(expected_values)
