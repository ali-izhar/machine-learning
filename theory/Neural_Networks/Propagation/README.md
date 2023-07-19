## Forward Propagation
Forward propagation computes the output of a neural network given inputs. Traditionally, it's done sequentially, layer by layer, which isn't suitable for parallel processing. To enable parallelization, we use matrix multiplication, termed as matrix forward propagation.

### Sequential Forward Propagation
```python
def dense(a_in, W, b, g):
    """
    Computes the output of a dense layer using inputs, weights, bias, and activation function.
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        z = np.dot(W[:,j], a_in) + b[j]
        a_out[j] = g(z)
    return a_out
```

### Matrix Forward Propagation
Stack input vectors into 2-D matrices to allow matrix multiplication, thereby enabling parallel computation.
```python
def dense(A_in, W, B, g):
    """
    Computes the output of a dense layer using inputs, weights, bias, and activation function.
    """
    Z = np.matmul(A_in, W) + B
    A_out = g(Z)
    return A_out
```

## Backward Propagation
Backward propagation computes the gradient of the loss function concerning the network's weights and biases.

### Sequential Backward Propagation
```python
def dense(dA, A_in, W, b, g_prime):
    """
    Computes the gradient of the loss function concerning weights and biases of a dense layer.
    """
    units = W.shape[1]
    dA_in, dW, db = np.zeros(A_in.shape), np.zeros(W.shape), np.zeros(units)
    for j in range(units):
        z = np.dot(W[:,j], A_in) + b[j]
        dA_in += dA[j] * W[:,j]
        dW[:,j] = dA[j] * A_in
        db[j] = dA[j] * g_prime(z)
    return (dA_in, dW, db)
```

### Matrix Backward Propagation
Stack input vectors into 2-D matrices to allow matrix multiplication and facilitate parallel computation.
```python
def dense(dA, A_in, W, B, g_prime):
    """
    Computes the gradient of the loss function concerning weights and biases of a dense layer.
    """
    dZ = dA * g_prime(A_in @ W + B)
    dA_in = dZ @ W.T
    dW = A_in.T @ dZ
    dB = np.sum(dZ, axis=0)
    return (dA_in, dW, dB)
```