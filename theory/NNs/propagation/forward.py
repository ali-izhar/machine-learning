import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dense(A_in, W, B, g):
    Z = np.matmul(A_in, W) + B
    A_out = g(Z)
    return A_out

def sequential(X, W, B, g):
    L = len(W)                      # number of layers
    A = [X]                         # list of activation values
    for l in range(L):              # for each layer
        a_in = A[l]                 # input activation values
        W_l = W[l]                  # weights for layer l
        b_l = B[l]                  # bias for layer l
        a_out = dense(a_in, W_l, b_l, g)
        A.append(a_out)
    return (A)


def predict(X, W, B, g):
    """
    Computes prediction for sequential NN
    Args:
      X (ndarray (n, )) : Input values, n features
      W (list (L, ))    : List of weight matrices, L layers
      B (list (L, ))    : List of bias
      g (function)      : Activation function
    Returns:
      y_pred (ndarray (j, )) : Prediction, j units
    """
    m = X.shape[0]                  # number of samples
    p = np.zeros((m, 1))            # prediction
    for i in range(m):              # for each sample
        x = X[i,:]                  # ith sample
        A = sequential(x, W, B, g)  # forward propagation
        y_pred = A[-1]              # prediction
        p[i] = y_pred               # store prediction
    return (p)