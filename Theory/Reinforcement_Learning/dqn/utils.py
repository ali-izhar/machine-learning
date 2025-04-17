import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()  # Turn on interactive mode for real-time plot updates


def plot_durations(episode_durations, show_result=False):
    """Plot episode durations and moving average

    Args:
        episode_durations: List of episode durations (number of steps before terminal state)
        show_result: If True, shows final results; if False, shows training progress

    Plotting both raw episode durations and a moving average helps visualize
    the trend while still seeing the variance in performance.
    """
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    # Moving average smooths out the noise to show learning trends
    if len(durations_t) >= 100:
        # Use a sliding window to calculate means for each window of 100 episodes
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        # Pad with zeros for first 99 episodes where we don't have 100 samples yet
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # Save the figure
    plt.savefig("episode_durations.png")

    # Update the plot
    plt.pause(0.001)  # Small pause to update plots
    if is_ipython:
        if not show_result:
            # For training visualization: clear previous plot and redraw
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            # For final results: just display the plot
            display.display(plt.gcf())
