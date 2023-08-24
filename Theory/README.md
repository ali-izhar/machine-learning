# Understanding the Basic Concept of Matrix Multiplication in ML
Matrix multiplication, often represented as $Y = W \times X + b$, is fundamental to the workings of neural networks and a wide variety of machine learning algorithms. In this article, we'll delve into matrix multiplication, emphasizing its role in machine learning.

## Delving into $W \times X + b$
In the realm of machine learning, we frequently deal with an input dataset $X$ and a corresponding output $Y$. The primary objective of a machine learning algorithm is to discern a function $f$ that can effectively map inputs $X$ to their outputs $Y$. This is mathematically captured as $f(X)=Y$. To ascertain $f$, we need to determine the optimal parameters (or weights) $W$ and bias $b$ such that the equation $f(X)=W \times X + b$ holds true. Here:

- $W$ is a matrix, representing the weights.
- $b$ is a vector, symbolizing the bias.

The dimensions of $W$ are influenced by the number of rows in $X$ (features) and the number of columns in $Y$ (output dimensions). Meanwhile, $b$'s size is set by the number of rows in $Y$.

## Generalized Representation
Given an input data $X$ and output data $Y$, we can represent these matrices in their generalized form:

$$
X = \begin{bmatrix}
| & | & \cdots & | \\
x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\
| & | & \cdots & | \\
\end{bmatrix}
$$

Where each $x^{(i)}$ is a column vector representing the $i$-th sample in the dataset and $m$ is the total number of samples.

$$
W = \begin{bmatrix}
-- & w^{(1)} & -- \\
-- & w^{(2)} & -- \\
\vdots & \vdots & \vdots \\
-- & w^{(n)} & -- \\
\end{bmatrix}
$$

Or, $W$ can also be represented as:

$$
W = \begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1n} \\
w_{21} & w_{22} & \cdots & w_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m1} & w_{m2} & \cdots & w_{mn} \\
\end{bmatrix}
$$

Here, $w_{ij}$ represents the weight of the connection between the $i$-th input and the $j$-th output node.

$$
b = \begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_n \\
\end{bmatrix}
$$

Where each $b_i$ is the bias for the $i$-th output node.

## Matrix Multiplication Unveiled
When executing the operation $W \times X + b$:

1. Matrix Multiplication (Dot Product): Each row of matrix $W$ is multiplied with each column of matrix $X$. This results in a new matrix where each entry is the sum of the products of the corresponding row and column entries.
2. Adding the Bias: The resulting matrix from the dot product is then added to the bias vector $b$. This operation is done element-wise, i.e., each entry in the matrix is added to its corresponding entry in the bias vector.

The outcome of this process is a new matrix, which provides the predicted outputs corresponding to the input data in $X$.

> Note: For the matrix multiplication to be valid, the number of columns in $W$ must match the number of rows in $X$. That's why in many scenarios, we see the transpose of $W$ (i.e., $W^T$) being used to align the matrices properly for multiplication.

## Matrix Dimensions in Neural Networks
Consider a neural network with the following architecture:

- **Input layer**: 2 nodes
- **First hidden layer**: 3 nodes
- **Second hidden layer**: 5 nodes
- **Third hidden layer**: 4 nodes
- **Fourth hidden layer**: 2 nodes
- **Output layer**: 1 node

To denote the number of nodes in each layer, we use the notation $n^{[l]}$. For example:
- $n^{[1]}$ represents the number of nodes in the first hidden layer.
- $n^{[2]}$ represents the number of nodes in the second hidden layer, and so on.

The output $Z^{[l]}$ of each layer $l$ is calculated using the formula:

$$
Z^{[l]} = W^{[l]} \times A^{[l-1]} + b^{[l]}
$$

Where:
- $Z^{[l]}$ is the output of the layer.
- $W^{[l]}$ is the weight matrix.
- $A^{[l-1]}$ is the input matrix.
- $b^{[l]}$ is the bias vector.

To understand the dimensions of the weight matrices $W$, we can use the following table:

| Layer Output $Z$ | Calculation | Dimensions | Dimensions of $W$ |
| :--------------: | :---------: | :--------: | :---------------: |
| $Z^{[1]}$        | $W^{[1]} \times X$ | $(3 \times 2) \times (2 \times 1) = 3 \times 1$ | $3 \times 2$ |
| $Z^{[2]}$        | $W^{[2]} \times A^{[1]}$ | $(5 \times 3) \times (3 \times 1) = 5 \times 1$ | $5 \times 3$ |
| $Z^{[3]}$        | $W^{[3]} \times A^{[2]}$ | $(4 \times 5) \times (5 \times 1) = 4 \times 1$ | $4 \times 5$ |
| $Z^{[4]}$        | $W^{[4]} \times A^{[3]}$ | $(2 \times 4) \times (4 \times 1) = 2 \times 1$ | $2 \times 4$ |
| $Z^{[5]}$        | $W^{[5]} \times A^{[4]}$ | $(1 \times 2) \times (2 \times 1) = 1 \times 1$ | $1 \times 2$ |

The dimensions of the bias vectors $b$ for each layer are:
- $b^{[1]}$ is a $3 \times 1$ matrix.
- $b^{[2]}$ is a $5 \times 1$ matrix.
- $b^{[3]}$ is a $4 \times 1$ matrix.
- $b^{[4]}$ is a $2 \times 1$ matrix.
- $b^{[5]}$ is a $1 \times 1$ matrix.