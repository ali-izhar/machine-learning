import numpy as np
from scipy.special import erfinv
from scipy.stats import binom

__all__ = ['inverse_cdf_gaussian', 'inverse_cdf_binomial']

def inverse_cdf_gaussian(y, mu, sigma):
    """
    Calculates the inverse of the CDF of the Gaussian distribution.
    :param y (float or ndarray): The probability or array of probabilities.
    :param mu (float): Mean of the Gaussian distribution.
    :param sigma (float): Standard deviation of the Gaussian distribution.
    :return x (float or ndarray): The corresponding value(s) from the Gaussian distribution that
    correspond to the given probability value(s).
    """
    x = mu + np.sqrt(2) * sigma * erfinv(2 * y - 1)
    return x

def inverse_cdf_binomial(y, n, p):
    """
    Calculates the inverse of the CDF of the binomial distribution.
    :param y (float or ndarray): The probability or array of probabilities.
    :param n (int): Number of trials in the binomial distribution.
    :param p (float): Probability of success in each trial.
    :return x (float or ndarray): The corresponding value(s) from the binomial distribution that
    correspond to the given probability value(s).
    """
    x = binom.ppf(y, n, p)
    return x