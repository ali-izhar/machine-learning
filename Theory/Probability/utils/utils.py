import numpy as np
import scipy
import matplotlib.pyplot as plt
from .generators import Generator
from .simulations import Dice

generator = Generator()

__all__ = ["plot_uniform", "plot_exponential", "plot_normal", "plot_gamma", "plot_beta"]


def plot_uniform(a, b):
    """Plots the PDF and CDF of a Uniform Distribution.
    a (float): Lower bound of the distribution
    b (float): Upper bound of the distribution
    """
    # Generating values for the uniform distribution
    x = np.linspace(a, b, 1000)
    y = [1 / (b - a) for _ in x]

    # Set up the figure and axes for a 2-column layout
    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plotting the Probability Density Function (PDF) on the first subplot
    axes[0].plot(x, y, label='PDF', color='blue')
    axes[0].fill_between(x, y, color='blue', alpha=0.3)
    axes[0].axhline(1 / (b - a), color='red', linestyle='--')
    axes[0].set_ylim(0, 1 / (b - a) + 0.05)
    axes[0].set_xlim(a, b)
    axes[0].set_title('Probability Density Function of a Uniform Distribution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True)

    # Plotting the Cumulative Distribution Function (CDF) on the second subplot
    y_cdf = np.linspace(0, 1, 1000)
    x_cdf = a + (b - a) * y_cdf
    axes[1].plot(x_cdf, y_cdf, label='CDF', color='green')
    axes[1].set_title('Cumulative Distribution Function of a Uniform Distribution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_xlim(a, b)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_exponential(lambdas):
    """Plots the PDF and CDF of an Exponential Distribution for multiple lambda values on a single graph.
    lambdas (list of float): List of rate parameters (lambda) of the distribution
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))
    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plotting PDF for each lambda
    for i, lambda_param in enumerate(lambdas):
        x = np.linspace(0, 10 / lambda_param, 1000)
        pdf = lambda_param * np.exp(-lambda_param * x)
        axes[0].plot(x, pdf, color=colors[i], label=f'λ = {lambda_param}')

    axes[0].set_title('Exponential Distribution PDFs')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True)

    # Plotting CDF for each lambda
    for i, lambda_param in enumerate(lambdas):
        x = np.linspace(0, 10 / lambda_param, 1000)
        cdf = 1 - np.exp(-lambda_param * x)
        axes[1].plot(x, cdf, color=colors[i], label=f'λ = {lambda_param}')

    axes[1].set_title('Exponential Distribution CDFs')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_normal(mus, sigmas):
    """Plots the PDFs and CDFs of Normal Distributions for multiple mean (mu) 
    and standard deviation (sigma) values.
    mus (list of float): List of means (mu) of the distributions
    sigmas (list of float): List of standard deviations (sigma) of the distributions
    """
    _, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(mus)))

    # Plotting PDF for each (mu, sigma)
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        pdf = scipy.stats.norm.pdf(x, mu, sigma)
        axes[0].plot(x, pdf, color=colors[i], label=f'μ = {mu}, σ = {sigma}')

    axes[0].set_title('Normal Distribution PDFs')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True)

    # Plotting CDF for each (mu, sigma)
    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        cdf = scipy.stats.norm.cdf(x, mu, sigma)
        axes[1].plot(x, cdf, color=colors[i], label=f'μ = {mu}, σ = {sigma}')

    axes[1].set_title('Normal Distribution CDFs')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_gamma(alpha, beta):
    """Plots the PDF and CDF of a Gamma Distribution.
    alpha (float): Shape parameter (alpha/k)
    beta (float): Rate parameter (beta/theta)
    """
    x = np.linspace(0, 20 / beta, 1000)
    pdf = (beta**alpha) * (x**(alpha-1)) * np.exp(-beta*x) / scipy.special.gamma(alpha)
    cdf = scipy.special.gammainc(alpha, beta * x)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, pdf, label='PDF', color='blue')
    plt.title('Gamma Distribution PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, cdf, label='CDF', color='green')
    plt.title('Gamma Distribution CDF')
    plt.xlabel('x')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_beta(alpha, beta):
    """
    Plots the PDF and CDF of a Beta Distribution.

    Parameters:
    alpha (float): Alpha parameter
    beta (float): Beta parameter
    """
    x = np.linspace(0, 1, 1000)
    pdf = scipy.stats.beta.pdf(x, alpha, beta)
    cdf = scipy.stats.beta.cdf(x, alpha, beta)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, pdf, label='PDF', color='blue')
    plt.title('Beta Distribution PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, cdf, label='CDF', color='green')
    plt.title('Beta Distribution CDF')
    plt.xlabel('x')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


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