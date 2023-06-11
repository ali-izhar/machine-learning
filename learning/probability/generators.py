import numpy as np
from scipy.special import erfinv
from scipy.stats import binom

__all__ = ['Generator']

class Generator(object):
    def __init__(self):
        pass

    def _inverse_cdf_gaussian(self, y, mu, sigma):
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
    
    def _inverse_cdf_binomial(self, y, n, p):
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

    def uniform_generator(self, a, b, num_samples=100):
        """
        Generates an array of uniformly distributed random numbers in the range [a, b).
        :param a (float): Lower bound of the range.
        :param b (float): Upper bound of the range.
        :param size (int): Number of samples to generate (default: 100).
        :return: Array of uniformly distributed random numbers.
        """
        np.random.seed(42)
        return np.random.uniform(a, b, num_samples)
    
    def gaussian_generator(self, mu, sigma, num_samples=100):
        """
        Generates an array of normally distributed random numbers with mean mu and standard
        deviation sigma.
        :param mu (float): Mean of the Gaussian distribution.
        :param sigma (float): Standard deviation of the Gaussian distribution.
        :param size (int): Number of samples to generate (default: 100).
        :return: Array of normally distributed random numbers.
        """

        # Generate an array of uniformly distributed random numbers in the range [0, 1)
        u = self.uniform_generator(0, 1, num_samples)

        # Use the inverse CDF technique to generate an array of normally distributed random numbers
        array = self._inverse_cdf_gaussian(u, mu, sigma)
        return array
    
    def binomial_generator(self, n, p, num_samples=100):
        """
        Generates an array of binomially distributed random numbers with n trials and probability of
        success p.
        :param n (int): Number of trials in the binomial distribution.
        :param p (float): Probability of success in each trial.
        :param size (int): Number of samples to generate (default: 100).
        :return: Array of binomially distributed random numbers.
        """

        # Generate an array of uniformly distributed random numbers in the range [0, 1)
        u = self.uniform_generator(0, 1, num_samples)

        # Use the inverse CDF technique to generate an array of binomially distributed random numbers
        array = self._inverse_cdf_binomial(u, n, p)
        return array