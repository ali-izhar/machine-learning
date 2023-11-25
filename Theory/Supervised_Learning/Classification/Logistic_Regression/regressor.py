import numpy as np
from sklearn.metrics import mean_squared_error

class LogisticRegressor:
    def __init__(self, learning_rate, num_iterations, regularization=False, lambda_=1.0, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.verbose = verbose
        self.w = None
        self.b = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """Compute the cost function for all the training samples"""
        z = np.dot(X, self.w) + self.b
        h = self.sigmoid(z)
        cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        return cost
    
    def compute_gradient(self, X, y):
        """Compute the gradient for all the training samples"""
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        h = self.sigmoid(z)
        dw = (1/m) * np.dot(X.T, (h - y))
        db = (1/m) * np.sum(h - y)
        if self.regularization:
            dw += (self.lambda_/m) * self.w
        return dw, db

    def fit(self, X, y):
        """Fit the model given the training data using gradient descent"""
        self.w = np.zeros((X.shape[1], 1))
        self.b = 0
        self.costs = []

        for i in range(self.num_iterations):
            dw, db = self.compute_gradient(X, y)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            cost = self.compute_cost(X, y)
            self.costs.append(cost)
            if self.verbose and i % 10 == 0:
                print(f'Cost after iteration {i}: {cost}')

    def predict(self, X):
        """Predict the class labels for the provided data"""
        z = np.dot(X, self.w) + self.b
        h = self.sigmoid(z)
        return (h > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """Evaluate the model using the provided data"""
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)