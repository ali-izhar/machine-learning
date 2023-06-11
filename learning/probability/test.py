from utils import *

def test_gaussian():
    gaussian_distributions = [
        (0, 1, 1000),
        (5, 3, 1000),
        (10, 5, 1000)
    ]
    plot_gaussian_distributions(gaussian_distributions)

def test_binomial():
    binomal_distributions = [
        (12, 0.4, 1000),
        (15, 0.5, 1000),
        (25, 0.8, 1000)
    ]
    plot_binomial_distributions(binomal_distributions)

if __name__ == '__main__':
    test_gaussian()
    test_binomial()