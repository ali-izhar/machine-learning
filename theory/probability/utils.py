import numpy as np
import matplotlib.pyplot as plt
from probability.generators import Generator
from probability.simulations import Dice

__all__ = ['plot_gaussian_distributions', 'plot_binomial_distributions', 'plot_dice_hist', 'plot_dice_stats']

generator = Generator()

def plot_gaussian_distributions(gaussian_distributions):
    """
    Plots the given Gaussian distributions as histograms on a single figure for comparison.
    :param gaussian_distributions: different Gaussian distributions to plot.
    """

    # Define the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot the Gaussian distributions
    for mu, sigma, num_samples in gaussian_distributions:
        # Generate an array of normally distributed random numbers
        array = generator.gaussian_generator(mu, sigma, num_samples)

        # Plot the histogram
        ax.hist(array, bins=100, density=True, alpha=0.5, label=f'N({mu}, {sigma})')

    # Set the title and axis labels
    ax.set_title('Histogram of Gaussian distributions')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequencies')

    # Set the legend
    ax.legend()

    # Show the plot
    plt.show()


def plot_binomial_distributions(binomal_distributions):
    """
    Plots the given Binomial distributions as histograms on a single figure for comparison.
    Just like the Gaussian distribution, make the discrete Binomial distribution continuous by
    connecting the discrete points with a line and filling the area under the line.
    :param binomal_distributions: different Binomial distributions to plot.
    """

    # Define the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot the Binomial distributions
    for n, p, num_samples in binomal_distributions:
        # Generate an array of binomially distributed random numbers
        array = generator.binomial_generator(n, p, num_samples)

        # Plot the histogram
        ax.hist(array, bins=100, density=True, alpha=0.5, label=f'B({n}, {p})')

    # Set the title and axis labels
    ax.set_title('Histogram of Binomial distributions')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequencies')

    # Set the legend
    ax.legend()

    # Show the plot
    plt.show()


def plot_dice_hist(sides, rolls, condition=None):
    """
    Plot a histogram of the roll results.
    Args:
    sides (int): The number of sides on the die.
    rolls (int): The number of rolls.
    condition (function, optional): A condition function that takes the result of a roll and returns a boolean.
    """
    dice = Dice(sides)
    rolls = dice.roll_many(rolls, condition)
    plt.hist(rolls, bins=np.arange(2 * dice.sides + 2), edgecolor='black', align='left')
    plt.show()


def plot_dice_stats(n_sides_range, n_rolls):
    """
    Plot the mean, variance, and covariance of the sum of dice rolls for a range of dice sides.
    Args:
    n_sides_range (range): A range of dice sides.
    n_rolls (int): The number of rolls.
    """
    means, vars, covs = [], [], []
    for n_sides in n_sides_range:
        dice = Dice(n_sides)
        rolls = dice.roll_many(n_rolls)
        sums = np.sum(rolls.reshape(-1, 2), axis=1)
        means.append(np.mean(sums))
        vars.append(np.var(sums))
        covs.append(np.cov(rolls[::2], rolls[1::2])[0, 1])

    plt.figure(figsize=(15,5))

    plt.subplot(131)
    plt.plot(n_sides_range, means, marker='o')
    plt.title('Mean of the sum')
    plt.xlabel('Number of sides')
    plt.ylabel('Mean')

    plt.subplot(132)
    plt.plot(n_sides_range, vars, marker='o')
    plt.title('Variance of the sum')
    plt.xlabel('Number of sides')
    plt.ylabel('Variance')

    plt.subplot(133)
    plt.plot(n_sides_range, covs, marker='o')
    plt.title('Covariance of the joint distribution')
    plt.xlabel('Number of sides')
    plt.ylabel('Covariance')

    plt.tight_layout()
    plt.show()