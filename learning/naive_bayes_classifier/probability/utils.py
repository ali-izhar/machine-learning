import matplotlib.pyplot as plt
from probability.generators import Generator

__all__ = ['plot_gaussian_distributions', 'plot_binomial_distributions']

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