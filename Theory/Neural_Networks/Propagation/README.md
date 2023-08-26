## Forward Propagation
Forward propagation computes the output of a neural network given inputs. Traditionally, it's done sequentially, layer by layer, which isn't suitable for parallel processing. To enable parallelization, we use matrix multiplication, termed as matrix forward propagation.

To compute forward propagation for layer $l$, we need the following:

- Inputs
    - Input vector $a^{[l-1]}$
    - Weight matrix $W^{[l]}$
    - Bias vector $b^{[l]}$
    - Activation function $g^{[l]}$
- Outputs
    - Output vector $a^{[l]}$
    - Cache $z^{[l]}$

The output of layer $l$ is computed as follows:

$$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$

$$a^{[l]} = g^{[l]}(z^{[l]})$$

When we compute forward propagation, we cache the intermediate values of $z^{[l]}$ and $a^{[l]}$ to use them in backward propagation.

## Backward Propagation
Backward propagation computes the gradient of the loss function concerning the network's weights and biases. To compute backward propagation for layer $l$, we need the following:

- Inputs
    - $da^{[l]}$
- Outputs
    - $da^{[l-1]}$
    - $dW^{[l]}$
    - $db^{[l]}$

To compute the gradient of the loss function concerning the weights and biases of layer $l$, we use the following formulas:

$$dz^{[l]} = da^{[l]} * g'^{[l]}(z^{[l]})$$

$$dW^{[l]} = dz^{[l]}a^{[l-1]T}$$

$$db^{[l]} = \sum_{i=1}^{m}dz^{[l_i]}$$

$$da^{[l-1]} = W^{[l]T}dz^{[l]}$$

## Derivations
Computing forward propagation is straightforward. However, computing backward propagation is a bit tricky. We need to derive the gradient of the loss function concerning the weights and biases of each layer. To do so, we use the chain rule.

### Forward Propagation Equations
1. $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$
2. $a^{[l]} = g^{[l]}(z^{[l]})$

### Backward Propagation Equations
#### Derivation of $z^{[l]}$ :
To compute the gradient of the loss function $L$ with respect to $z^{[l]}$, we use the chain rule as follows:

$$\frac{\partial L}{\partial z^{[l]}} = \frac{\partial L}{\partial a^{[l]}} \frac{\partial a^{[l]}}{\partial z^{[l]}}$$

In the above equation, we can compute $\frac{\partial L}{\partial a^{[l]}}$ from the previous layer. However, we need to compute $\frac{\partial a^{[l]}}{\partial z^{[l]}}$ which is the derivative of the activation function $g^{[l]}$. Therefore:

$$\frac{\partial L}{\partial z^{[l]}} = \frac{\partial L}{\partial a^{[l]}} \frac{\partial a^{[l]}}{\partial z^{[l]}} = \frac{\partial L}{\partial a^{[l]}} g'^{[l]}(z^{[l]}) = da^{[l]} * g'^{[l]}(z^{[l]})$$

#### Derivation of $W^{[l]}$ :
To compute the gradient of the loss function $L$ with respect to $W^{[l]}$, we use the chain rule as follows:

$$\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial z^{[l]}} \frac{\partial z^{[l]}}{\partial W^{[l]}}$$

In the above equation, we can compute $\frac{\partial L}{\partial z^{[l]}}$ from the previous layer. However, we need to compute $\frac{\partial z^{[l]}}{\partial W^{[l]}}$ which is the derivative of the forward propagation equation $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$. Therefore:

$$\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial z^{[l]}} \frac{\partial z^{[l]}}{\partial W^{[l]}} = \frac{\partial L}{\partial z^{[l]}} a^{[l-1]T} = dz^{[l]}a^{[l-1]T}$$

#### Derivation of $b^{[l]}$ :
To compute the gradient of the loss function $L$ with respect to $b^{[l]}$, we use the chain rule as follows:

$$\frac{\partial L}{\partial b^{[l]}} = \frac{\partial L}{\partial z^{[l]}} \frac{\partial z^{[l]}}{\partial b^{[l]}}$$

In the above equation, we can compute $\frac{\partial L}{\partial z^{[l]}}$ from the previous layer. However, we need to compute $\frac{\partial z^{[l]}}{\partial b^{[l]}}$ which is the derivative of the forward propagation equation $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$. Therefore:

$$\frac{\partial L}{\partial b^{[l]}} = \frac{\partial L}{\partial z^{[l]}} \frac{\partial z^{[l]}}{\partial b^{[l]}} = \frac{\partial L}{\partial z^{[l]}} = dz^{[l]}$$

#### Derivation of $a^{[l-1]}$ :
To compute the gradient of the loss function $L$ with respect to $a^{[l-1]}$, we use the chain rule as follows:

$$\frac{\partial L}{\partial a^{[l-1]}} = \frac{\partial L}{\partial z^{[l]}} \frac{\partial z^{[l]}}{\partial a^{[l-1]}}$$

In the above equation, we can compute $\frac{\partial L}{\partial z^{[l]}}$ from the previous layer. However, we need to compute $\frac{\partial z^{[l]}}{\partial a^{[l-1]}}$ which is the derivative of the forward propagation equation $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$. Therefore:

$$\frac{\partial L}{\partial a^{[l-1]}} = \frac{\partial L}{\partial z^{[l]}} \frac{\partial z^{[l]}}{\partial a^{[l-1]}} = \frac{\partial L}{\partial z^{[l]}} W^{[l]T} = da^{[l]} W^{[l]T}$$

## Implementation
We implement forward and backward propagation using sequential and matrix approaches.

```python
### Sequential Forward Propagation
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

### Matrix Forward Propagation
def dense(A_in, W, B, g):
    """
    Computes the output of a dense layer using inputs, weights, bias, and activation function.
    """
    Z = np.matmul(A_in, W) + B
    A_out = g(Z)
    return A_out
```

```python
### Sequential Backward Propagation
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

### Matrix Backward Propagation
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