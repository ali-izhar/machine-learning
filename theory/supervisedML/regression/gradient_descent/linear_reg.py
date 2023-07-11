import numpy as np
from typing import List, Tuple

__all__ = ['compute_cost', 'compute_gradient', 'gradient_descent', 'predict']

def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Computes the squared error cost function for linear regression.
    """
    m = X.shape[0]
    cost = (1/(2*m)) * np.sum((np.dot(X, w) + b - y)**2)
    return cost
    

def compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, 
                     regularization: bool = False, lambda_: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Computes the gradient of the squared error cost function for linear regression.
    """
    m = X.shape[0]
    dw = (1/m) * np.dot(X.T, (np.dot(X, w) + b - y))
    db = (1/m) * np.sum(np.dot(X, w) + b - y)
    if regularization:
        dw += (lambda_/m) * w
    return dw, db


def gradient_descent(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, learning_rate: float, num_iterations: int, verbose=False) -> Tuple[np.ndarray, float, List[float]]:
    """
    Performs gradient descent to learn w and b
    """
    cost_history = []
    params_history = []
    for i in range(num_iterations):
        dw, db = compute_gradient(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        params_history.append((w, b))
        if verbose and i % (num_iterations // 10) == 0:
            print(f"Iteration {i}: Cost {round(cost, 3)}")
    return w, b, cost_history, params_history


def predict(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Predicts the labels for the data x given the parameters w and b.
    """
    y_pred = np.dot(x, w) + b
    return y_pred


def stochastic_gradient_descent(x: np.ndarray, y: np.ndarray, w: float, b: float, learning_rate: float, num_iterations: int) -> Tuple[float, float, List[float]]:
    """
    Performs stochastic gradient descent to learn w and b
    """
    cost_history = []
    m = x.shape[0]
    for _ in range(num_iterations):
        for i in range(m):
            dw, db = compute_gradient(np.array([x[i]]), np.array([y[i]]), w, b)
            w = w - learning_rate * dw
            b = b - learning_rate * db
        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)
    return w, b, cost_history


def minibatch_gradient_descent(x: np.ndarray, y: np.ndarray, w: float, b: float, learning_rate: float, num_iterations: int, batch_size: int) -> Tuple[float, float, List[float]]:
    """
    Performs mini-batch gradient descent to learn w and b
    """
    cost_history = []
    m = x.shape[0]
    for _ in range(num_iterations):
        for i in range(0, m, batch_size):
            end = i + batch_size if i + batch_size <= m else m
            dw, db = compute_gradient(x[i:end], y[i:end], w, b)
            w = w - learning_rate * dw
            b = b - learning_rate * db
        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)
    return w, b, cost_history