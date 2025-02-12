import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from thompson import ThompsonSamplingAgent, DistributionType
from scipy import stats


def run_bandit_experiment(
    true_probabilities: list,
    num_trials: int,
    num_episodes: int,
    dist_type: DistributionType = DistributionType.BETA,
    sigma: float = 1.0,
) -> tuple[float, float, list]:
    """
    Run multi-armed bandit experiment with Thompson Sampling.

    Args:
        true_probabilities (list): True success probability for each action
        num_trials (int): Number of trials per episode
        num_episodes (int): Number of episodes to run
        dist_type (DistributionType): Type of probability distribution to use
        sigma (float): Standard deviation for normal distribution

    Returns:
        tuple[float, float, list]: (avg_return, std_return, action_counts)
    """
    num_actions = len(true_probabilities)
    returns = []
    total_action_counts = np.zeros(num_actions)

    for _ in range(num_episodes):
        # Initialize agent with specified distribution
        agent = ThompsonSamplingAgent(num_actions, dist_type=dist_type, sigma=sigma)
        episode_return = 0
        action_counts = np.zeros(num_actions)

        # Run one episode
        for _ in range(num_trials):
            # Select action using Thompson Sampling
            action = agent.select_action()
            action_counts[action] += 1

            # Generate reward
            reward = 1.0 if np.random.random() < true_probabilities[action] else 0.0
            episode_return += reward

            # Update agent's knowledge
            agent.update(action, reward)

        returns.append(episode_return)
        total_action_counts += action_counts

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    avg_action_counts = total_action_counts / num_episodes

    return avg_return, std_return, avg_action_counts


class BanditVisualizer:
    """Visualizes the evolution of probability distributions in Thompson Sampling."""

    def __init__(
        self,
        true_probabilities: list,
        dist_type: DistributionType = DistributionType.BETA,
        num_samples: int = 500,  # Further reduced sample size
    ):
        self.true_probs = true_probabilities
        self.num_actions = len(true_probabilities)
        self.dist_type = dist_type
        self.num_samples = num_samples

        # Initialize agent
        self.agent = ThompsonSamplingAgent(self.num_actions, dist_type=dist_type)

        # Setup plot
        plt.style.use("seaborn-v0_8")
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.axes = self.axes.flatten()

        # Pre-compute x values
        if dist_type == DistributionType.BETA:
            self.x = np.linspace(0, 1, 100)  # Reduced resolution
        else:
            self.x = np.linspace(-0.5, 1.5, 100)

        # Initialize plot elements
        self.lines = []
        self.texts = []

        # Setup static elements once
        for i, ax in enumerate(self.axes):
            ax.axvline(true_probabilities[i], color="r", linestyle="--", alpha=0.5)
            (line,) = ax.plot([], [], "b-", lw=2)
            text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

            self.lines.append(line)
            self.texts.append(text)

            ax.set_title(f"Action {i+1} (p={true_probabilities[i]:.2f})")
            ax.set_xlim(self.x[0], self.x[-1])
            ax.set_ylim(0, 5)
            ax.grid(True, alpha=0.3)

        self.action_counts = np.zeros(self.num_actions)
        self.rewards = np.zeros(self.num_actions)
        plt.tight_layout()

    def get_beta_pdf(self, a, b):
        """Compute Beta PDF directly without sampling."""
        return stats.beta.pdf(self.x, a, b)

    def get_normal_pdf(self, mu, sigma):
        """Compute Normal PDF directly without sampling."""
        return stats.norm.pdf(self.x, mu, sigma)

    def update(self, frame):
        """Update visualization for each frame."""
        # Select action and get reward
        action = self.agent.select_action()
        reward = 1.0 if np.random.random() < self.true_probs[action] else 0.0

        # Update agent and counts
        self.agent.update(action, reward)
        self.action_counts[action] += 1
        self.rewards[action] += reward

        # Update plots
        plot_elements = []
        for i in range(self.num_actions):
            # Compute PDF directly instead of KDE
            if self.dist_type == DistributionType.BETA:
                density = self.get_beta_pdf(
                    self.agent.successes[i], self.agent.failures[i]
                )
            else:
                density = self.get_normal_pdf(
                    self.agent.means[i],
                    self.agent.sigma / np.sqrt(self.agent.counts[i] + 1),
                )

            # Update line
            self.lines[i].set_data(self.x, density)

            # Update text
            success_rate = self.rewards[i] / max(1, self.action_counts[i])
            self.texts[i].set_text(
                f"N={self.action_counts[i]:.0f}\nRate={success_rate:.2f}"
            )

            plot_elements.extend([self.lines[i], self.texts[i]])

        return tuple(plot_elements)

    def animate(self, num_trials: int):
        """Create and display animation."""
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=num_trials,
            interval=20,
            blit=True,
            cache_frame_data=False,
        )
        plt.show()


if __name__ == "__main__":
    # Problem setup
    TRUE_PROBS = [0.3, 0.5, 0.2, 0.8]  # M1: 30%, M2: 50%, M3: 20%, M4: 80%
    NUM_TRIALS = 100
    NUM_EPISODES = 1000
    OPTIMAL_RETURN = max(TRUE_PROBS) * NUM_TRIALS

    # Test both distributions
    for dist_type in DistributionType:
        # Run experiment
        avg_return, std_return, action_counts = run_bandit_experiment(
            true_probabilities=TRUE_PROBS,
            num_trials=NUM_TRIALS,
            num_episodes=NUM_EPISODES,
            dist_type=dist_type,
            sigma=0.5,  # Smaller sigma for more focused normal sampling
        )

        # Print results
        print(f"\nThompson Sampling with {dist_type.value} distribution")
        print("-" * 45)
        print(f"Machine Probabilities: M1={30}%, M2={50}%, M3={20}%, M4={80}%")
        print(f"Number of Episodes: {NUM_EPISODES}")
        print(f"Number of Trials per Episode: {NUM_TRIALS}")

        print("\nResults:")
        print(f"Expected Return: {avg_return:.2f} ± {std_return:.2f}")
        print(f"Optimal Return: {OPTIMAL_RETURN:.2f}")
        print(f"Performance Ratio: {(avg_return/OPTIMAL_RETURN)*100:.1f}%")

        print("\nAverage Action Selection Counts:")
        for i, count in enumerate(action_counts):
            print(f"Machine {i+1} ({TRUE_PROBS[i]*100:>3.0f}%): {count:>6.1f} times")

    # Run visualization for both distribution types
    for dist_type in DistributionType:
        visualizer = BanditVisualizer(TRUE_PROBS, dist_type)
        visualizer.animate(NUM_TRIALS)
