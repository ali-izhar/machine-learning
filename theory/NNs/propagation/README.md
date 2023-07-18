# Model Training Steps

### 1. Model Architecture
Specify how to compute the output of the model for a given input. Consider the following neural network model with 2 hidden layers (25 and 15 neurons) and 1 output layer (1 neuron):

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])
```

### 2. Loss and Cost Functions
Specify a loss function to measure how well the model fits the data. In binary classification, the loss function is called the `logistic loss` which, in statistics, is called the `binary cross-entropy` loss function. The binary cross-entropy loss function is given by:

```python
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss=BinaryCrossentropy())
```

For a regression problem, we can specify the `mean squared error` loss function:

```python
from tensorflow.keras.losses import MeanSquaredError
model.compile(loss=MeanSquaredError())
```

### 3. Gradient Descent
Specify an optimization algorithm to minimize the loss function. The most common optimization algorithm is `gradient descent`. In order to compute the gradient of the loss function with respect to the weights and biases of the model, we need to use the `backpropagation` algorithm. The backpropagation algorithm is an efficient way to compute the gradient of the loss function with respect to the weights and biases of a neural network. The backpropagation algorithm is given by:

```python
model.fit(X, y, epochs=10)
```

## Forward Propagation
The forward propagation algorithm is used to calculate the output of a neural network given a set of inputs. The traditional implementation of forward propagation is a sequential process, where the output of one layer is fed as input to the next layer. This sequential process is not suitable for parallelization, which is a requirement for training large neural networks. The forward propagation algorithm can be modified to allow for parallelization by using matrix multiplication. This is known as matrix forward propagation.

### Sequential Forward Propagation
```python
def dense(a_in, W, b, g):
    """
    Calculates the output of a dense layer given the input, weights, bias, and activation function.
    """
    units = W.shape[1]              # j units
    a_out = np.zeros(units)         # j units
    for j in range(units):          # for each unit
        w = W[:,j]                  # weights for jth unit
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)             # activation of jth unit
    return (a_out)
```

### Matrix Forward Propagation
Notice that all the input vectors are stacked into 2-D matrices. This allows for matrix multiplication to be used to calculate the output of each layer. The output of each layer is a 2-D matrix, where each row is the output of a single input vector. This allows for parallelization of the forward propagation algorithm.

```python
def dense(A_in, W, B, g):
    """
    Calculates the output of a dense layer given the input, weights, bias, and activation function.
    """
    Z = np.matmul(A_in, W) + B
    A_out = g(Z)
    return A_out
```

## Backward Propagation
The backward propagation algorithm is used to calculate the gradient of the loss function with respect to the weights and biases of a neural network. 

### Sequential Backward Propagation
```python
def dense_backward(dA, A_in, W, b, g_prime):
    """
    Calculates the gradient of the loss function with respect to the weights and biases of a dense layer.
    """
    units = W.shape[1]              # j units
    dA_in = np.zeros(A_in.shape)    # i units
    dW = np.zeros(W.shape)          # i x j weights
    db = np.zeros(units)            # j biases
    for j in range(units):          # for each unit
        w = W[:,j]                  # weights for jth unit
        z = np.dot(w, A_in) + b[j]
        dA_in += dA[j] * w
        dW[:,j] = dA[j] * A_in
        db[j] = dA[j] * g_prime(z)
    return (dA_in, dW, db)
```

### Matrix Backward Propagation
Notice that all the input vectors are stacked into 2-D matrices. This allows for matrix multiplication to be used to calculate the gradient of the loss function with respect to the weights and biases of each layer. The gradient of the loss function with respect to the weights and biases of each layer is a 2-D matrix, where each row is the gradient of the loss function with respect to the weights and biases of a single input vector. This allows for parallelization of the backward propagation algorithm.

```python
def dense_backward(dA, A_in, W, B, g_prime):
    """
    Calculates the gradient of the loss function with respect to the weights and biases of a dense layer.
    """
    dZ = dA * g_prime(A_in @ W + B)
    dA_in = dZ @ W.T
    dW = A_in.T @ dZ
    dB = np.sum(dZ, axis=0)
    return (dA_in, dW, dB)
```
