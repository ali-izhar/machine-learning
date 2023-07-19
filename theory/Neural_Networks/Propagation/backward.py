# backward propagation

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dense_backward(dA, AT, W, B, g):
    """
    Perform the backward propagation step for a dense (fully connected) layer.

    Args:
        dA: Gradient of the cost with respect to the activation matrix A.
        AT: Input matrix A with shape (m, n_prev).
        W: Weight matrix with shape (n_prev, n).
        B: Bias vector with shape (1, n).
        g: Activation function.

    Returns:
        dAT: Gradient of the cost with respect to the input matrix A with shape (m, n_prev).
        dW: Gradient of the cost with respect to the weight matrix with shape (n_prev, n).
        dB: Gradient of the cost with respect to the bias vector with shape (1, n).
    """
    Z = np.dot(AT, W) + B
    dZ = dA * g(Z) * (1 - g(Z))
    dAT = np.dot(dZ, W.T)
    dW = np.dot(AT.T, dZ)
    dB = np.sum(dZ, axis=0, keepdims=True)
    return dAT, dW, dB

def sequential_backward(dA, X, W, B, g):
    """
    Perform backward propagation through a sequential neural network.

    Args:
        dA: Gradient of the cost with respect to the activation matrix A.
        X: Input matrix with shape (m, n_0).
        W: List of weight matrices.
        B: List of bias vectors.
        g: Activation function.

    Returns:
        dX: Gradient of the cost with respect to the input matrix X with shape (m, n_0).
        dW: List of gradients of the cost with respect to the weight matrices.
        dB: List of gradients of the cost with respect to the bias vectors.
    """
    dX = dA
    dW = []
    dB = []
    for i in reversed(range(len(W))):
        dX, dWi, dBi = dense_backward(dX, X, W[i], B[i], g)
        dW.insert(0, dWi)
        dB.insert(0, dBi)
        X = X
    return dX, dW, dB

def update_parameters(W, B, dW, dB, alpha):
    """
    Update the parameters of the network.

    Args:
        W: List of weight matrices.
        B: List of bias vectors.
        dW: List of gradients of the cost with respect to the weight matrices.
        dB: List of gradients of the cost with respect to the bias vectors.
        alpha: Learning rate.

    Returns:
        W: Updated list of weight matrices.
        B: Updated list of bias vectors.
    """
    for i in range(len(W)):
        W[i] -= alpha * dW[i]
        B[i] -= alpha * dB[i]
    return W, B

def train(X, Y, W, B, g, alpha, epochs):
    """
    Train the neural network.

    Args:
        X: Input matrix with shape (m, n_0).
        Y: Output matrix with shape (m, n_L).
        W: List of weight matrices.
        B: List of bias vectors.
        g: Activation function.
        alpha: Learning rate.
        epochs: Number of epochs to train for.

    Returns:
        W: List of weight matrices.
        B: List of bias vectors.
        costs: List of costs over the training epochs.
    """
    costs = []
    for i in range(epochs):
        A = sequential(X, W, B, g)
        cost = cross_entropy(A, Y)
        costs.append(cost)
        dA = cross_entropy_backward(A, Y)
        dX, dW, dB = sequential_backward(dA, X, W, B, g)
        W, B = update_parameters(W, B, dW, dB, alpha)
    return W, B, costs

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
        Z = np.dot(A, W[i]) + B[i]
        A = g(Z)
    return A
