# dp_policy_evaluation.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

plt.style.use("seaborn-v0_8")
W = LinearSegmentedColormap.from_list("w", ["w", "w"], N=256)

# Define possible actions as directional movements
ACTIONS = {
    0: [1, 0],  # north
    1: [-1, 0],  # south
    2: [0, -1],  # west
    3: [0, 1],  # east
}


class GridWorld:
    def __init__(self, size=4):
        """
        A gridworld environment with absorbing states at [0, 0] and [size - 1, size - 1].

        Args:
            size (int): the dimension of the grid in each direction
        """
        self.state_value = np.zeros((size, size))
        self.fig, self.ax = None, None
        self.heatmap = None

    def reset(self):
        """Reset the state values to zero."""
        self.state_value = np.zeros_like(self.state_value)

    def step(self, state, action):
        """
        Take a step in the environment given current state and action.

        Args:
            state (tuple): Current state coordinates (x, y)
            action (list): Movement direction [dx, dy]

        Returns:
            tuple: (next_state, reward)
        """
        # Check if current state is terminal
        size = len(self.state_value) - 1
        if (state == (0, 0)) or (state == (size, size)):
            return state, 0

        # Calculate next state
        s_1 = (state[0] + action[0], state[1] + action[1])
        reward = -1

        # Handle boundary conditions
        if s_1[0] < 0 or s_1[0] >= len(self.state_value):
            s_1 = state  # Bounce back if hitting north-south walls
        elif s_1[1] < 0 or s_1[1] >= len(self.state_value):
            s_1 = state  # Bounce back if hitting east-west walls

        return s_1, reward

    def setup_plot(self):
        """Initialize the plot for animation."""
        size = min(len(self.state_value), 20)
        self.fig, self.ax = plt.subplots(figsize=(size, size))
        self.ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
        self.heatmap = sn.heatmap(
            self.state_value,
            annot=True,
            fmt=".1f",
            cmap=W,
            linewidths=1,
            linecolor="black",
            cbar=False,
            ax=self.ax,
        )
        return self.fig

    def update_plot(self, iteration):
        """Update the plot with current state values."""
        self.ax.clear()
        self.ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
        self.ax.set_title(f"Value Function after {iteration} iterations")

        # Update heatmap
        sn.heatmap(
            self.state_value,
            annot=True,
            fmt=".1f",
            cmap=W,
            linewidths=1,
            linecolor="black",
            cbar=False,
            ax=self.ax,
        )

        # Keep axis labels visible
        self.ax.set_xlabel("Column")
        self.ax.set_ylabel("Row")

    def bellman_expectation(self, state, probs, discount):
        """
        Makes a one step lookahead and applies the bellman expectation equation.

        Args:
            state (tuple): Current state coordinates (x, y)
            probs (list): Transition probabilities for each action
            discount (float): Discount factor

        Returns:
            float: New value for the specified state
        """
        value = 0
        for action_idx, action in ACTIONS.items():
            next_state, reward = self.step(state, action)
            value += probs[action_idx] * (
                reward + discount * self.state_value[next_state]
            )
        return value


def policy_evaluation_animated(
    env, policy=None, max_iterations=1000, discount=1.0, in_place=False
):
    """
    Animated policy evaluation using dynamic programming.
    """
    if policy is None:
        policy = np.ones((*env.state_value.shape, len(ACTIONS))) * 0.25

    fig = env.setup_plot()

    def update(iteration):
        # Cache old values if not updating in-place
        values = env.state_value if in_place else np.empty_like(env.state_value)

        # Update each state's value
        for i in range(len(env.state_value)):
            for j in range(len(env.state_value[i])):
                state = (i, j)
                value = env.bellman_expectation(state, policy[i, j], discount)
                values[i, j] = value * discount

        # Update value table
        env.state_value = values
        env.update_plot(iteration)

    # Create animation with specific frames
    frames = [1, 2, 3, 1000]
    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000,  # 1 second between frames
        repeat=False,
        blit=False,  # Set to False to ensure proper updates
    )

    # Keep a reference to prevent garbage collection
    plt.show(block=True)
    return env.state_value


if __name__ == "__main__":
    env = GridWorld(4)
    final_values = policy_evaluation_animated(
        env, policy=None, max_iterations=1, discount=1.0, in_place=False
    )
