import numpy as np
from scipy.special import erfinv

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