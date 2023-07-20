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

## Gradient Descent
Gradient descent is an optimization algorithm used in machine learning and deep learning models to minimize a function iteratively. It's frequently used to find the optimal solution to many problems.

## What is Gradient Descent?
Gradient descent is a first-order iterative optimization algorithm for finding a minimum of a function. To find a local minimum, the function steps in the direction of the negative of the gradient. In the context of machine learning, this function is typically a loss function that measures the discrepancy between the model's predictions and the actual data. By minimizing this loss function, we obtain the parameters that result in the best model performance.

## Math Behind Gradient Descent
The objective of gradient descent is to minimize the cost function $J(w, b)$ where $w$ and $b$ represent the parameters of our model and $J(w, b)$ is defined as:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

where $m$ is the number of training examples, $\hat{y}^{(i)}$ is the predicted value for the $i^{th}$ training example, and $y^{(i)}$ is the actual value for the $i^{th}$ training example.

The gradient descent algorithm updates each parameters iteratively by taking steps proportional to the negative of the gradient of the cost function with respect to that parameter, computed at the current point:

$$w := w - \alpha \frac{\partial J(w, b)}{\partial w}$$

$$b := b - \alpha \frac{\partial J(w, b)}{\partial b}$$

where $\alpha$ is the learning rate. The learning rate determines the size of the steps taken during gradient descent. If the learning rate is too small, the algorithm will take a long time to converge. If the learning rate is too large, the algorithm may never converge.

> Note: The gradient descent algorithm updates the parameters simultaneously, therefore, the parameters are updated at the same time. The correct implementation is as follows:
>
> $$w_{temp} = w - \alpha \frac{\partial J(w, b)}{\partial w}$$
>
> $$b_{temp} = b - \alpha \frac{\partial J(w, b)}{\partial b}$$
>
> $$w = w_{temp}$$
>
> $$b = b_{temp}$$

## Algorithm
The gradient descent algorithm is as follows:

1. Initialize the parameters $w$ and $b$ to 0.
2. Compute the gradient of the cost function with respect to $w$ and $b$.
3. Update the parameters $w$ and $b$ using the following equations:
    - $w := w - \alpha \frac{\partial J(w, b)}{\partial w}$
    - $b := b - \alpha \frac{\partial J(w, b)}{\partial b}$
4. Repeat steps 2 and 3 until the cost function converges.

The partial derivatives of the cost function with respect to $w$ and $b$ are as follows:

$$\frac{\partial J(w, b)}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})x^{(i)}$$

$$\frac{\partial J(w, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$