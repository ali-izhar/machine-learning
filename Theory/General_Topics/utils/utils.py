import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_ewma']

# Generate synthetic temperature data
np.random.seed(0)

def compute_ewma(temperatures, beta, correct_bias=True):
    v = 0  # Initialize v
    ewma = [0]  # Start with the first value as 0 for consistency
    for t, theta in enumerate(temperatures, 1):
        v = beta * v + (1 - beta) * theta
        if correct_bias:
            v_corrected = v / (1 - beta**t)  # Apply bias correction
        else:
            v_corrected = v
        ewma.append(v_corrected)
    return ewma[1:]  # Exclude the initial dummy value


def plot_ewma(temperatures, betas, correct_bias=True):
    colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, label='Actual Temperatures', color='black', linewidth=3, alpha=0.3)
    for beta, color in zip(betas, colors):
        temperatures_ewma = compute_ewma(temperatures, beta, correct_bias)

        if beta == 0.9:
            plt.plot(temperatures_ewma, label=f'EWMA (beta={beta})', color='red', linewidth=2)
        else:
            plt.plot(temperatures_ewma, label=f'EWMA (beta={beta})', color=color)

    plt.legend()

    if correct_bias:
        plt.title('Exponentially Weighted Moving Average of Temperatures (Corrected Bias)')
    else:
        plt.title('Exponentially Weighted Moving Average of Temperatures (Uncorrected Bias)')
    
    plt.xlabel('Day')
    plt.ylabel('Temperature')
    plt.show()