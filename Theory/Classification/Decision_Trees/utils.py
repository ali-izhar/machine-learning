import numpy as np
import matplotlib.pyplot as plt

from .basics import entropy, weighted_entropy, information_gain

def plot_entropy():
    x = np.linspace(0, 1, 100)
    y = [entropy(p) for p in x]
    plt.plot(x, y)
    plt.xlabel('p')
    plt.ylabel('H(p)')
    plt.show()


def plot_weighted_entropy():
    x = np.linspace(0, 1, 100)
    y = [weighted_entropy(x, [1] * 100, [1] * 50, [1] * 50) for x in x]
    plt.plot(x, y)
    plt.xlabel('p')
    plt.ylabel('H(p)')
    plt.show()


def plot_information_gain():
    x = np.linspace(0, 1, 100)
    y = [information_gain(x, [1] * 100, [1] * 50, [1] * 50) for x in x]
    plt.plot(x, y)
    plt.xlabel('p')
    plt.ylabel('IG(p)')
    plt.show()