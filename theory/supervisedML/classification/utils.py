import numpy as np
import matplotlib.pyplot as plt

def plot_sigmoid(interval: np.ndarray, size: list = [10, 4]):

    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=size)
    sigmoid_values = _sigmoid(interval)
    ax.plot(interval, sigmoid_values, label='sigmoid function')

    # color left and right from z=0 differently
    positive = interval >= 0
    ax.fill_between(interval, sigmoid_values, where=positive, color='blue', alpha=0.3, label='z >= 0')
    ax.fill_between(interval, sigmoid_values, where=~positive, color='red', alpha=0.3, label='z < 0')

    # draw threshold line at z=0
    ax.axvline(0, color='black', linestyle='--', label='z = 0')
    ax.set_xlabel('z')
    ax.set_ylabel('sigmoid(z)')
    ax.set_title('Sigmoid function with decision boundary')
    ax.legend()
    plt.show()