import numpy as np
from typing import List, Tuple
from sklearn.metrics import mean_squared_error

__all__ = ["LinearRegressor"]

class LinearRegressor:
    def __init__(self, learning_rate: float, num_iterations: int, verbose=False, regularization: bool = False, lambda_: float = 1.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.w = None
        self.b = None
        self.regularization = regularization
        self.lambda_ = lambda_

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        m = X.shape[0]
        cost = (1/(2*m)) * np.sum((np.dot(X, self.w) + self.b - y)**2)
        return cost

    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        m = X.shape[0]
        dw = (1/m) * np.dot(X.T, (np.dot(X, self.w) + self.b - y))
        db = (1/m) * np.sum(np.dot(X, self.w) + self.b - y)
        if self.regularization:
            dw += (self.lambda_/m) * self.w
        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
        self.w = np.zeros(X.shape[1])
        self.b = 0
        cost_history = []
        params_history = []
        for i in range(self.num_iterations):
            dw, db = self.compute_gradient(X, y)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            cost = self.compute_cost(X, y)
            cost_history.append(cost)
            params_history.append((self.w, self.b))
            if self.verbose and i % (self.num_iterations // 10) == 0:
                print(f"Iteration {i}: Cost {round(cost, 3)}")
        return self.w, self.b, params_history, cost_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)
    
    def stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[np.ndarray, float, List[float]]:
        cost_history = []
        m = X.shape[0]
        for _ in range(self.num_iterations):
            for i in range(0, m, batch_size):
                end = i + batch_size if i + batch_size <= m else m
                dw, db = self.compute_gradient(X[i:end], y[i:end])
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
            cost = self.compute_cost(X, y)
            cost_history.append(cost)
        return self.w, self.b, cost_history
    
    def minibatch_gradient_descent(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> Tuple[np.ndarray, float, List[float]]:
        cost_history = []
        m = X.shape[0]
        for _ in range(self.num_iterations):
            for i in range(0, m, batch_size):
                end = i + batch_size if i + batch_size <= m else m
                dw, db = self.compute_gradient(X[i:end], y[i:end])
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
            cost = self.compute_cost(X, y)
            cost_history.append(cost)
        return self.w, self.b, cost_history