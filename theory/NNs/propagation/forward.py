import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dense(AT, W, B, g):
    """
    Perform the dense (fully connected) layer operation.

    Args:
        AT: Input matrix A with shape (m, n_prev).
        W: Weight matrix with shape (n_prev, n).
        B: Bias vector with shape (1, n).
        g: Activation function.

    Returns:
        A_out: Output matrix A with shape (m, n).
    """
    Z = np.dot(AT, W) + B
    A_out = g(Z)
    return A_out

def sequential(X, W, B, g):
    """
    Perform forward propagation through a sequential neural network.

    Args:
        X: Input matrix with shape (m, n_0).
        W: List of weight matrices.
        B: List of bias vectors.
        g: Activation function.

    Returns:
        A: Output matrix with shape (m, n_L).
    """
    A = X
    for i in range(len(W)):
        # A is the most recent activation matrix
        Z = np.dot(A, W[i]) + B[i]
        A = g(Z)
    return A

def predict(X, W, B, g):
    """
    Predict the class labels for input examples.

    Args:
        X: Input matrix with shape (m, n_0).
        W: List of weight matrices.
        B: List of bias vectors.
        g: Activation function.

    Returns:
        predictions: Array of predicted class labels.
    """
    A = sequential(X, W, B, g)
    # Predict the class with the highest probability
    predictions = np.argmax(A, axis=1)
    return predictions