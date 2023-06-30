import numpy as np
from matplotlib import pyplot as plt
from typing import List
from .linear_regression_gd import predict, compute_cost

__all__ = ['plot_data', 'plot_cost_history', 'plot_data_and_cost', 'plot_data_and_predictions', 'plot_contour']

def plot_data(x: np.ndarray, y: np.ndarray, w: float, b: float) -> None:
    """
    Plots the data and the linear regression line.
    """
    plt.scatter(x, y)
    plt.plot(x, predict(x, w, b), color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data')
    plt.show()


def plot_cost_history(cost_history: List[float]) -> None:
    """
    Plots the cost history.
    """
    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost history')
    plt.show()


def plot_data_and_cost(x: np.ndarray, y: np.ndarray, w: float, b: float, cost_history: List[float]) -> None:
    """
    Plots the data and the linear regression line.
    """
    plot_data(x, y, w, b)
    plot_cost_history(cost_history)


def plot_data_and_predictions(x: np.ndarray, y: np.ndarray, w: float, b: float, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plots the data and the linear regression line.
    """
    plot_data(x, y, w, b)
    plt.scatter(x_test, y_test, color='green')
    plt.plot(x_test, predict(x_test, w, b), color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and predictions')
    plt.show()


def plot_contour(x: np.ndarray, y: np.ndarray, params_history: List[float], zoom: bool = False) -> None:
    """
    Plots the cost history and the parameters history in the contour plot.
    """

    w_history = [params[0] for params in params_history]
    b_history = [params[1] for params in params_history]

    if zoom:
        w_values = np.arange(180, 220, 0.5)
        b_values = np.arange(80, 120, 0.5)
        levels = [1, 5, 10, 20]
        # Only take the last 50 values for the zoomed plot
        w_history = w_history[-50:]
        b_history = b_history[-50:]
    else:
        w_values = np.linspace(-10, 10, 100)
        b_values = np.linspace(-10, 10, 100)
        levels = np.logspace(-2, 3, 30)
    
    W, B = np.meshgrid(w_values, b_values)
    costs = np.zeros((W.shape[0], W.shape[1]))
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            costs[i, j] = compute_cost(x, y, W[i, j], B[i, j])
    
    plt.figure(figsize=(10, 8))
    contours = plt.contour(W, B, costs, levels=levels, cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.scatter(w_history, b_history, color='red')
    plt.plot(w_history, b_history, color='red', linewidth=2)
    plt.xlabel('w')
    plt.ylabel('b')
    plt.title('Contour plot of cost function')
    plt.show()

    return x, y, w_history, b_history