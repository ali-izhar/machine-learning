import numpy as np

class Generator(object):
    def __init__(self, seed=42):
        self.seed = seed

    def uniform_generator(self, a, b, num_samples=100):
        """
        Generates an array of uniformly distributed random numbers in the range [a, b).
        :param a (float): Lower bound of the range.
        :param b (float): Upper bound of the range.
        :param size (int): Number of samples to generate (default: 100).
        :return: Array of uniformly distributed random numbers.
        """
        np.random.seed(self.seed)
        return np.random.uniform(a, b, num_samples)
    
