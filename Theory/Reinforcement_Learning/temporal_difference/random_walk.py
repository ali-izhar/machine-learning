"""
Random Walk TD Learning Implementation
===================================

This is an implementation of the Random Walk example from Sutton & Barto's Chapter 6 on TD Learning,
modified to use a biased probability distribution.

Environment Description:
----------------------
- 7 states: [0, 1, 2, 3, 4, 5, 6] representing [T_L, A, B, C, D, E, T_R]
- All episodes start in center state (C = 3)
- Biased probability: 20% of going left, 80% of going right
- Terminal states: 0 (T_L) and 6 (T_R)
- Reward structure: 
  * +1 when reaching right terminal state (T_R)
  * 0 for all other transitions

True Values
-----------
The true state values represent the probability of reaching the right terminal state:

Using the Bellman equation for each state with P(left)=0.2 and P(right)=0.8:
For each non-terminal state: V(s) = 0.2*V(s-1) + 0.8*V(s+1)

For a biased random walk with absorbing barriers, the probability of absorption at the
right barrier starting from position i is:
V(i) = (1 - (P(left)/P(right))^i) / (1 - (P(left)/P(right))^(n+1))

Where n=5 is the number of non-terminal states. With P(left)/P(right) = 0.25:

- State A (1): (1 - 0.25^1) / (1 - 0.25^6) ≈ 0.7501
- State B (2): (1 - 0.25^2) / (1 - 0.25^6) ≈ 0.9377
- State C (3): (1 - 0.25^3) / (1 - 0.25^6) ≈ 0.9846
- State D (4): (1 - 0.25^4) / (1 - 0.25^6) ≈ 0.9963
- State E (5): (1 - 0.25^5) / (1 - 0.25^6) ≈ 0.9992
- Terminal states: T_L (0) = 0.0, T_R (6) = 1.0

These values represent "What's the probability of eventually reaching the right terminal state if starting from this position?"

With the strong 80% rightward bias:
1. From state E: You're one step away from T_R with an 80% chance of going directly there.
   Even if you first move left, you'll likely get back to T_R eventually. Thus, V(E) ≈ 0.999 (almost certain).

2. From state D: Still very likely to reach T_R due to the rightward bias, V(D) ≈ 0.996.

3. From state C (middle): Even from the center, you're much more likely to drift right than left,
   so V(C) ≈ 0.984 (compared to 0.5 in the unbiased 50/50 case).

4. From state B: Despite being closer to the left terminal, the rightward bias gives you
   a high chance of reaching T_R, V(B) ≈ 0.938.

5. From state A: Even from the leftmost non-terminal state, you're still three times more likely
   to reach T_R than T_L due to the bias. V(A) ≈ 0.75.

Experiment Configurations:
------------------------
The code reproduces two key experiments from the book:
1. Example 6.2: Compares value estimation and RMS error for TD vs MC
2. Figure 6.2: Demonstrates batch updating performance

Learning Parameters:
------------------
- TD learning rates (α): [0.15, 0.1, 0.05]
- MC learning rates (α): [0.01, 0.02, 0.03, 0.04]
- Batch learning rate: 0.001
- Episodes: 100
- Runs: 100 (for averaging)
"""

import numpy as np
import matplotlib
from typing import List, Tuple

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# 0 is the left terminal state
# 6 is the right terminal state
# 1 ... 5 represents A ... E
VALUES = np.zeros(7)
VALUES[1:6] = 0.5
# For convenience, we assume all rewards are 0
# and the left terminal state has value 0, the right terminal state has value 1
# This trick has been used in Gambler's Problem
VALUES[6] = 1

# set up true state values for biased random walk (20% left, 80% right)
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:6] = [
    0.7501,
    0.9377,
    0.9846,
    0.9963,
    0.9992,
]
TRUE_VALUE[6] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1


def temporal_difference(values, alpha=0.1, batch=False):
    """
    Run one episode of TD(0) learning in the Random Walk environment.

    The episode starts from the center state (C) and continues until reaching
    a terminal state. For non-batch mode, updates state values using TD(0) rule:
    V(s) ← V(s) + α[R + V(s') - V(s)]

    Environment dynamics:
    - Start at center state (C = 3)
    - 20% probability of moving left, 80% probability of moving right
    - Terminal states: T_L (left) and T_R (right)
    - Reward: +1 only when reaching T_R, 0 otherwise

    Args:
        values: numpy array of current state values (length 7)
        alpha: learning rate (step size parameter)
        batch: if True, don't update values but return trajectory and rewards

    Returns:
        tuple (trajectory, rewards):
            trajectory: list of states visited
            rewards: list of rewards received
    """
    state = 3  # Start at center state
    trajectory = [state]
    rewards = [0]
    while True:
        old_state = state
        # 20% probability of moving left, 80% right
        if np.random.random() < 0.2:
            state -= 1
        else:
            state += 1

        # Get reward (+1 only when reaching right terminal)
        reward = 1.0 if state == 6 else 0.0
        trajectory.append(state)

        # TD update
        if not batch:
            # V(s) ← V(s) + α[R + V(s') - V(s)]
            values[old_state] += alpha * (reward + values[state] - values[old_state])
        if state == 6 or state == 0:  # Terminal states
            break
        rewards.append(reward)
    return trajectory, rewards


def monte_carlo(values, alpha=0.1, batch=False):
    """
    Run one episode of Monte Carlo learning in the Random Walk environment.

    The episode starts from the center state (C) and continues until reaching
    a terminal state. For non-batch mode, updates state values using MC rule:
    V(s) ← V(s) + α[G - V(s)] where G is the actual return.

    Environment dynamics:
    - Start at center state (C = 3)
    - 20% probability of moving left, 80% probability of moving right
    - Terminal states: T_L (left) and T_R (right)
    - Reward: +1 only when reaching T_R, 0 otherwise

    Args:
        values: numpy array of current state values (length 7)
        alpha: learning rate (step size parameter)
        batch: if True, don't update values but return trajectory and returns

    Returns:
        tuple (trajectory, returns):
            trajectory: list of states visited
            returns: list of returns (1.0 if reached T_R, 0.0 if reached T_L)
    """
    state = 3  # Start at center state
    trajectory = [3]

    # Generate trajectory until terminal state
    while True:
        # 20% probability of moving left, 80% right
        if np.random.random() < 0.2:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 6:  # Right terminal
            returns = 1.0
            break
        elif state == 0:  # Left terminal
            returns = 0.0
            break

    # Update values for all states in trajectory (except terminal)
    if not batch:
        for state_ in trajectory[:-1]:
            # V(s) ← V(s) + α[G - V(s)]
            values[state_] += alpha * (returns - values[state_])
    return trajectory, [returns] * (len(trajectory) - 1)


def plot_value_evolution():
    """
    Plot the evolution of state values over episodes (Example 6.2 left plot).

    Shows how TD(0) estimates evolve from initial values (0.5) towards true values
    after 0, 1, 10, and 100 episodes. Also plots true values for comparison.
    """
    episodes = [0, 1, 10, 100]
    current_values = np.copy(VALUES)
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(current_values, label=str(i) + " episodes")
        temporal_difference(current_values)
    plt.plot(TRUE_VALUE, label="true values")
    plt.xlabel("state")
    plt.ylabel("estimated value")
    plt.legend()


def compare_td_mc_errors():
    """
    Compare TD and MC methods with different learning rates (Example 6.2 right plot).

    Runs both TD and MC methods for 100 episodes with different learning rates:
    - TD: α ∈ [0.15, 0.1, 0.05]
    - MC: α ∈ [0.01, 0.02, 0.03, 0.04]

    Plots RMS error vs episodes averaged over 100 independent runs.
    """
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 100 + 1
    runs = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = "TD"
            linestyle = "solid"
        else:
            method = "MC"
            linestyle = "dashdot"
        for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(VALUES)
            for i in range(0, episodes):
                errors.append(
                    np.sqrt(np.sum(np.power(TRUE_VALUE - current_values, 2)) / 5.0)
                )
                if method == "TD":
                    temporal_difference(current_values, alpha=alpha)
                else:
                    monte_carlo(current_values, alpha=alpha)
            total_errors += np.asarray(errors)
        total_errors /= runs
        plt.plot(
            total_errors,
            linestyle=linestyle,
            label=method + ", alpha = %.02f" % (alpha),
        )
    plt.xlabel("episodes")
    plt.ylabel("RMS")
    plt.legend()


def batch_updating(method, episodes, alpha=0.001):
    """
    Implement batch updating for TD or MC methods (Figure 6.2).

    In batch updating, we:
    1. Generate multiple episodes
    2. Store all trajectories and rewards
    3. Repeatedly update values using all data until convergence
    4. Track RMS error after each episode

    Args:
        method: 'TD' or 'MC'
        episodes: number of episodes to run
        alpha: learning rate for batch updates

    Returns:
        numpy array of RMS errors averaged over 100 independent runs
    """
    runs = 100
    total_errors = np.zeros(episodes)
    for r in tqdm(range(0, runs)):
        current_values = np.copy(VALUES)
        errors = []
        trajectories = []
        rewards = []
        for ep in range(episodes):
            if method == "TD":
                trajectory_, rewards_ = temporal_difference(current_values, batch=True)
            else:
                trajectory_, rewards_ = monte_carlo(current_values, batch=True)
            trajectories.append(trajectory_)
            rewards.append(rewards_)
            while True:
                # Update values using all trajectories until convergence
                updates = np.zeros(7)
                for trajectory_, rewards_ in zip(trajectories, rewards):
                    for i in range(0, len(trajectory_) - 1):
                        if method == "TD":
                            updates[trajectory_[i]] += (
                                rewards_[i]
                                + current_values[trajectory_[i + 1]]
                                - current_values[trajectory_[i]]
                            )
                        else:
                            updates[trajectory_[i]] += (
                                rewards_[i] - current_values[trajectory_[i]]
                            )
                updates *= alpha
                if np.sum(np.abs(updates)) < 1e-3:  # Convergence check
                    break
                current_values += updates
            errors.append(
                np.sqrt(np.sum(np.power(current_values - TRUE_VALUE, 2)) / 5.0)
            )
        total_errors += np.asarray(errors)
    return total_errors / runs


def plot_example_6_2():
    """
    Create and save the plots for Example 6.2 from the book.

    Generates two subplots:
    1. Value function evolution over episodes
    2. RMS error comparison between TD and MC methods
    """
    # Create figure with more balanced dimensions
    fig = plt.figure(figsize=(12, 10))

    # Use GridSpec for more control over subplot layout
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # Value evolution subplot
    ax1 = fig.add_subplot(gs[0])
    plt.sca(ax1)
    plot_value_evolution()

    # Add annotation explaining value evolution
    value_explanation = """
    Each line shows state values after N episodes of TD learning.
    With 80% rightward bias, values converge rapidly toward the
    true values (probability of reaching T_R), with states E and D
    developing much higher values than in the unbiased case.
    """
    ax1.annotate(
        value_explanation,
        xy=(0.5, 0.05),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # TD vs MC error comparison subplot
    ax2 = fig.add_subplot(gs[1])
    plt.sca(ax2)
    compare_td_mc_errors()

    # Add annotation explaining TD vs MC comparison
    error_explanation = """
    Comparison of TD and MC learning methods with various learning rates.
    TD methods (solid lines) initially perform worse than MC methods (dash-dot)
    in the biased random walk environment. MC methods with higher learning 
    rates (α=0.03, 0.04) demonstrate better RMS error characteristics.
    """
    ax2.annotate(
        error_explanation,
        xy=(0.5, 0.05),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # Add overall title
    fig.suptitle(
        "Biased Random Walk (20% Left, 80% Right): TD vs MC Learning",
        fontsize=14,
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle

    plt.savefig("example_6_2.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_figure_6_2():
    """
    Create and save the plot for Figure 6.2 from the book.

    Compares the performance of batch TD and MC methods by plotting
    their RMS errors over episodes.
    """
    episodes = 100 + 1
    td_errors = batch_updating("TD", episodes)
    mc_errors = batch_updating("MC", episodes)

    plt.plot(td_errors, label="TD")
    plt.plot(mc_errors, label="MC")
    plt.xlabel("episodes")
    plt.ylabel("RMS error")
    plt.legend()

    plt.savefig("figure_6_2.png")
    plt.close()


def run_episode_batch(
    values: np.ndarray,
    method: str = "TD",
    batch: bool = False,
    alpha: float = 0.1,
    gamma: float = 1.0,
) -> Tuple[List[int], List[float]]:
    """
    Run one episode with batch updating support.

    Args:
        env: Random Walk environment
        values: Current state values as numpy array
        method: 'TD' or 'MC'
        batch: Whether to perform batch updating
        alpha: Learning rate
        gamma: Discount factor

    Returns:
        Tuple of (trajectory, rewards)
    """
    state = 3  # Start at center state
    trajectory = [state]
    rewards = []

    while True:
        old_state = state
        # Random move with 20% probability of moving left, 80% right
        if np.random.random() < 0.2:
            state -= 1
        else:
            state += 1

        # Get reward (+1 only when reaching right terminal)
        reward = 1.0 if state == 6 else 0.0
        rewards.append(reward)
        trajectory.append(state)

        if not batch and method == "TD":
            # TD update
            next_value = values[state] if state not in [0, 6] else reward
            if old_state not in [0, 6]:  # Only update non-terminal states
                values[old_state] += alpha * (
                    reward + gamma * next_value - values[old_state]
                )

        if state in [0, 6]:  # Terminal states
            break

    return trajectory, rewards


if __name__ == "__main__":
    plot_example_6_2()
    plot_figure_6_2()
