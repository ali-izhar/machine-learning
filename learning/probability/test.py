from utils import *

if __name__ == '__main__':
    # Define the Gaussian distributions to plot
    gaussian_distributions = [
        (0, 1, 1000),
        (5, 3, 1000),
        (10, 5, 1000)
    ]

    # Plot the Gaussian distributions
    plot_gaussian_distributions(gaussian_distributions)