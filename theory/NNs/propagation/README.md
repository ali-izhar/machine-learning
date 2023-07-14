# Forward Propagation
The forward propagation algorithm is used to calculate the output of a neural network given a set of inputs. The traditional implementation of forward propagation is a sequential process, where the output of one layer is fed as input to the next layer. This sequential process is not suitable for parallelization, which is a requirement for training large neural networks. The forward propagation algorithm can be modified to allow for parallelization by using matrix multiplication. This is known as matrix forward propagation.

## Sequential Forward Propagation
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

## Matrix Forward Propagation
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