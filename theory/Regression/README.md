# Linear Regression
Linear regression is a supervised learning algorithm used for predicting a continuous target variable based on one or more predictor variables. The goal is to find the best fit line that captures the relationship between the independent (predictor) and dependent (target) variables.

Linear regression can be categorized into two types based on the number of predictor variables used: simple linear regression (one predictor) and multiple linear regression (more than one predictor).

## Loss Function (Cost Function)
A loss function measures the difference between the model's predictions and the actual values. In linear regression, we typically use the Mean Squared Error (MSE) as the loss function, defined as:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

This is called the `squared error cost function` or `mean squared error (MSE)`. The goal of training a model is to find the optimal values for the parameters that minimize the loss function. That is $\text{minimize } J(w, b)$.

## Simple Linear Regression
In simple linear regression, we have one predictor variable. The function $f$ is defined as:

$$\hat{y}^{(i)} = f(x^{(i)}) = w x^{(i)} + b$$

Here, $\hat{y}^{(i)}$ is the predicted value, $x^{(i)}$ is the predictor variable, $w$ is the weight or coefficient, and $b$ is the bias or intercept. The weight and bias are the parameters of the model, which are learned during the training process.

## Multiple Linear Regression
In multiple linear regression, we have multiple predictor variables. The function $f$ then takes the form:

$$\hat{y}^{(i)} = f(x^{(i)}) = w_1 x_1^{(i)} + w_2 x_2^{(i)} + ... + w_n x_n^{(i)} + b$$

Here, $x_1^{(i)}, x_2^{(i)}, ..., x_n^{(i)}$ are the predictor variables and $w_1, w_2, ..., w_n$ are their respective weights. We aim to find the optimal values for these parameters that minimize the loss function.


## Vector Notation and Vectorization
In the previous sections, we expressed the linear regression model and multiple linear regression model as summations of the product of weights and features, and included an additional bias term:

$$\hat{y}^{(i)} = f(x^{(i)}) = w_1 x_1^{(i)} + w_2 x_2^{(i)} + ... + w_n x_n^{(i)} + b$$

In a non-vectorized implementation, this could be expressed in code as:

```python
def predict(x, w, b):
    y_hat = 0
    for i in range(len(x)):
        y_hat += w[i] * x[i]
    y_hat += b
    return y_hat
```

However, this form isn't very computationally efficient, especially for large datasets and high-dimensional feature spaces. It requires iterating over each feature individually to compute the summation.

To improve efficiency, we can utilize the concept of `vectorization.` Vectorization is a powerful concept in linear algebra and data science that allows operations to be performed on entire arrays (vectors, matrices) instead of individual elements. This leverages low-level optimizations and parallelism that lead to significant speed improvements.

In Python, we can use the NumPy library to perform vectorized operations. The previous function can be rewritten in a vectorized form as:

```python
def predict(x, w, b):
    y_hat = np.dot(w, x) + b
    return y_hat
```

In this vectorized version, `np.dot(w, x)` computes the dot product of the weight vector `w` and the feature vector `x`, effectively performing the summation of the product of weights and features.

## Expressing the Model in Vector Notation
For convenience and to express our models more succinctly, we often use vector notation. Let's define the weight vector $W=[w_1, w_2, ..., w_n]$ and the feature vector for a given sample $X^{(i)}=[x_1^{(i)}, x_2^{(i)}, ..., x_n^{(i)}]$. Then, we can express the model as:

$$\hat{y}^{(i)} = f(x^{(i)}) = W X^{(i)T} + b$$

Here, $X^{(i)T}$ denotes the transpose of the feature vector, making it a column vector and aligning it for the dot product operation with the row vector of weights $W$. This compact form is equivalent to the previous summation. The goal remains to minimize the loss function $J(W, b)$ in order to find the optimal parameters for our model.

## Performing Linear Regression with Normal Equation
The `normal equation` is a closed-form solution for finding the optimal parameters of a linear regression model. It is given by:

$$W = (X^T X)^{-1} X^T y$$

Here, $X$ is the feature matrix, $y$ is the target vector, and $W$ is the weight vector. The normal equation can be derived by setting the gradient of the loss function $J(W)$ to zero and solving for $W$.

The normal equation is computationally efficient for small datasets, but it is not suitable for large datasets because the matrix $X^T X$ is a square matrix of size $n \times n$, where $n$ is the number of features. The computational complexity of inverting such a matrix is $O(n^3)$. For large datasets, we can use gradient descent to find the optimal parameters.