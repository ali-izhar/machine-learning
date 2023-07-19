import numpy as np
from scipy.special import comb

__all__ = ['Dist']

class Dist(object):
    def __init__(self):
        pass

    def pdf_uniform(self, x, a, b):
        """
        Calculates the probability density function of the uniform distribution
        between the given range [a, b) at a given value x.
        :param x (float or ndarray): The value(s) at which to calculate the PDF.
        :param a (float): Lower bound of the range.
        :param b (float): Upper bound of the range.
        :return p (float or ndarray): The probability density value(s) for the given x value(s).
        """
        pdf = 1 / (b - a) if a <= x <= b else 0
        return pdf
    
    def pdf_gaussian(self, x, mu, sigma):
        """
        Calculates the probability density function of the Gaussian distribution 
        at a given value x.
        :param x (float or ndarray): The value(s) at which to calculate the PDF.
        :param mu (float): Mean of the Gaussian distribution.
        :param sigma (float): Standard deviation of the Gaussian distribution.
        :return p (float or ndarray): The probability density value(s) for the given x value(s).
        """
        pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return pdf
    
    def pdf_binomial(self, x, n, p):
        """
        Calculates the probability density function of the binomial distribution
        at a given value x.
        :param x (float or ndarray): The value(s) at which to calculate the PDF.
        :param n (int): Number of trials in the binomial distribution.
        :param p (float): Probability of success in each trial.
        :return p (float or ndarray): The probability density value(s) for the given x value(s).
        """
        pdf = comb(n, x) * p ** x * (1 - p) ** (n - x)
        return pdf