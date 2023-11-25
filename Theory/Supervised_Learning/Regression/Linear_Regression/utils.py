import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List
from regressor import LinearRegressor


def plot_scatter_data(X: np.ndarray, Y: np.ndarray, size=(8, 6)) -> None:
    plt.style.use('_mpl-gallery')
    plt.figure(figsize=(size[0], size[1]))
    plt.scatter(X, Y)
    plt.show()
    return None

def plot_scatter_with_best_fit(X: np.ndarray, Y: np.ndarray, model: LinearRegressor, size=(8, 6)) -> None:
    plt.figure(figsize=(size[0], size[1]))
    plt.scatter(X, Y, label='Actual')

    Y_pred = model.predict(X)
    mse = model.evaluate(X, Y)

    plt.plot(X, Y_pred, color='red', label=f'Predicted (y = {model.w:.2f}x + {model.b:.2f}, MSE = {mse:.2f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    return None

def plot_cost_history(cost_history: List[float], x_label='Iterations', y_label='Cost', title='Cost history') -> None:
    plt.plot(cost_history, alpha=0.3)
    plt.plot(pd.Series(cost_history).rolling(window=int(len(cost_history)*0.02)+1).mean()) # for smoothed line
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    return None

def plot_contour(x: np.ndarray, y: np.ndarray, model: LinearRegressor):
    w_history = [params[0] for params in model.params_history]
    b_history = [params[1] for params in model.params_history]

    w1, w2 = min(w_history), w_history[-1]
    b1, b2 = min(b_history), b_history[-1]

    min_w = min(w2 - ((w2 - w1) / 10), w2)
    max_w = w2 + 20

    min_b = min(b2 - ((b2 - b1) / 8) - 10, b2)
    max_b = b2 + 20

    w_values = np.arange(min_w, max_w, 0.5)
    b_values = np.arange(min_b, max_b, 0.5)
    levels = [1, 5, 10, 20]

    distances = [np.sqrt((w_history[i] - w_history[i-1])**2 + (b_history[i] - b_history[i-1])**2) for i in range(1, len(w_history))]

    start = next(i for i, distance in enumerate(distances) if distance < 0.2)  # Adjust the threshold as necessary

    nth = (len(w_history) - start) // 20  # Adjust the denominator to change the frequency of arrows
    arrow_history = list(zip(w_history[start::nth], b_history[start::nth])) + [(w_history[-1], b_history[-1])]

    W, B = np.meshgrid(w_values, b_values)
    costs = np.zeros((W.shape[0], W.shape[1]))

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            costs[i, j] = model.compute_cost(x, y, W[i, j], B[i, j])

    plt.figure(figsize=(12, 4))
    contours = plt.contour(W, B, costs, levels=levels, cmap='viridis', alpha=0.7)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

    for i in range(1, len(arrow_history)):
        plt.annotate('', xy=arrow_history[i], xytext=arrow_history[i-1],
                     arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                     va='center', ha='center')

    plt.scatter(*zip(*arrow_history), color='red')
    x_line = plt.axvline(x=w2, color='black', linestyle='--')
    y_line = plt.axhline(y=b2, color='black', linestyle='--')
    plt.legend([x_line, y_line], [f'w = {w2:.2f}', f'b = {b2:.2f}'])
    plt.xlabel('w')
    plt.ylabel('b')
    plt.title('Contour plot of cost J(w, b), vs. b,w with path of gradient descent')
    plt.show()
    return None


if __name__ == "__main__":
    X = np.array([1, 2, 3, 4, 5, 6])
    Y = np.array([2, 3, 4, 5, 6, 7])
    model = LinearRegressor(learning_rate=0.01, num_iterations=1000, verbose=True)
    model.fit(X, Y)

    plot_scatter_data(X, Y)
    plot_scatter_with_best_fit(X, Y, model)
    plot_cost_history(model.cost_history)
    plot_contour(X, Y, model)