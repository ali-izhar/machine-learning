# Multiple Linear Regression
Recall that in regression problems we are given a set of $n$ labeled examples: $D=\{(x_1,y_1),\ldots,(x_n,y_n)\}$, where $x_i$ is an $m$-dimensional vector containing the features of the $i$-th example and $y_i$ is the label of the $i$-th example.

In linear regression problems, we assume that there is a linear relationship between the feature vector $x$ and the label $y$, so our model hypothesis takes the following form:

$$h_(x) = W^t x = \sum_{i=1}^m w_i x_i$$

where $W$ is a vector of weights of size $m$. Our goal is to find the parameters $w$ of this model that will minimize the sum of squared residuals:

$$J(w) = \sum_{i=1}^n (h(x_i) - y_i)^2 = \sum_{i=1}^n (w^t x_i - y_i)^2$$

Previously, we saw how to find the optimal $w$ for the simple case of $m=1$ using the normal equation. We will now extend this to the case of multiple features.

To simplify the derivation of the normal equations for the general case, we first define a matrix $X$ that contains all the feature vectors of the training set including the intercept terms:

$$X = \begin{bmatrix}
1 & x_{11} & x_{12} & \ldots & x_{1m} \\
1 & x_{21} & x_{22} & \ldots & x_{2m} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \ldots & x_{nm} \\
\end{bmatrix}$$

This matrix is called the design matrix. Each row in the design matrix represents an individual sample, and the columns represent the explanatory variables. The dimensions of the matrix are $n \times (m + 1)$, where $n$ is the number of samples and $m$ is the number of features.

In addition, we define the vector $y$ as an $n$-dimensional vector that contains all the target values:

$$y = \begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n \\
\end{bmatrix}$$

These definitions allow us to write the least squares cost function in the following matrix form:

$$J(w) = \sum_{i=1}^n (w^t x_i - y_i)^2 = (Xw - y)^t (Xw - y)$$

## Closed Form Solution
The gradient of $J(w)$ with respect to $w$ is given by:

$$\nabla_w J(w) = 2 X^t (Xw - y)$$

Setting the gradient to zero and solving for $w$ gives us the following closed form solution:

$$w* = (X^t X)^{-1} X^t y$$

Note that we assumed here that the columns of $X$ are linearly independent (i.e., $X$ has a full column rank), otherwise $X^tX$ is not invertible, and there is no unique solution for $w*$.

When the columns of $X$ are linearly dependent, we call this phenomenon **multicollinearity**. Mathematically, a set of variables is perfectly multicollinear if for all samples $i$:

$$\lambda_0 + \lambda_1 x_{i1} + \lambda_2 x_{i2} + \ldots + \lambda_m x_{im} = 0$$

where $\lambda_0, \lambda_1, \ldots, \lambda_m$ are constants, not all zero. In this case, the matrix $X^tX$ is not invertible, and there is no unique solution for $w*$.