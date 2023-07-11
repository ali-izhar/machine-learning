import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b):
    """
    Compute the cost function for logistic regression.
    """
    z = np.dot(X, w) + b
    h = sigmoid(z)
    cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    return cost

def compute_gradient(X, y, w, b, regularization=False, lambda_=1.0):
    """
    Compute the gradient of the cost function for logistic regression.
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    h = sigmoid(z)
    dw = (1/m) * np.dot(X.T, (h - y))
    db = (1/m) * np.sum(h - y)
    if regularization:
        dw += (lambda_/m) * w
    return dw, db

def gradient_descent(X, y, w, b, learning_rate, num_iterations, verbose=False):
    """
    Perform gradient descent for logistic regression.
    """
    costs = []
    for _ in range(num_iterations):
        dw, db = compute_gradient(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        cost = compute_cost(X, y, w, b)
        costs.append(cost)
        if verbose and _ % 10 == 0:
            print(f'Cost after iteration {_}: {cost[-1]}')
    return w, b, costs