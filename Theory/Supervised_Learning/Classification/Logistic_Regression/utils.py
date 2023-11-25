import numpy as np
import matplotlib.pyplot as plt
from .regressor import LogisticRegressor

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

def plot_decision_boundary(X: np.ndarray, y: np.ndarray, clf, size: list = [10, 4]):
    """
    Plot decision boundary of a classifier clf.
    """
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Make predictions across the grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=size)
    
    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.8, cmap='cividis')
    # Plot the points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='plasma', alpha=0.5, edgecolor='k', label='data points')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Decision boundary')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.show()


if __name__ == '__main__':
    interval = np.linspace(-10, 10, 100)
    plot_sigmoid(interval)

    X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)
    
    log_clf = LogisticRegressor(0.01, 1000, True, 0.01, True)
    log_clf.fit(X, y)

    plot_decision_boundary(X, y, log_clf)