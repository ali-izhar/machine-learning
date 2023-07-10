# Gradient Descent
Gradient descent is an optimization algorithm used in machine learning and deep learning models to minimize a function iteratively. It's frequently used to find the optimal solution to many problems.

## Table of Contents
1. [What is Gradient Descent?](#what-is-gradient-descent)
2. [Math Behind Gradient Descent](#math-behind-gradient-descent)
3. [Types of Gradient Descent](#types-of-gradient-descent)
    - [Batch Gradient Descent](#batch-gradient-descent)
    - [Stochastic Gradient Descent](#stochastic-gradient-descent)
    - [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)

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

## Types of Gradient Descent
There are three types of gradient descent: batch gradient descent, stochastic gradient descent, and mini-batch gradient descent.

### Batch Gradient Descent
Batch gradient descent is the most common type of gradient descent. It uses the entire training dataset to compute the gradient of the cost function. The gradient is computed by taking the average of the gradients of each training example in the dataset. Batch gradient descent is guaranteed to converge to the global minimum for convex error surfaces and to a local minimum for non-convex error surfaces. However, it is computationally expensive to compute the gradient of the cost function for each training example in the dataset.

### Stochastic Gradient Descent
Stochastic gradient descent (SGD) is an iterative method for optimizing an objective function with suitable smoothness properties. It uses a single training example to compute the gradient of the cost function. SGD is computationally less expensive than batch gradient descent because it only uses a single training example to compute the gradient of the cost function. However, the gradient computed using a single training example is very noisy and fluctuates a lot. This causes the cost function to fluctuate as well. As a result, SGD will not converge to the global minimum, but will instead wander around it. However, SGD is guaranteed to converge to a local minimum for convex error surfaces and to a local minimum for non-convex error surfaces.

### Mini-Batch Gradient Descent
Mini-batch gradient descent is a variation of stochastic gradient descent where instead of using a single training example to compute the gradient of the cost function, it uses a mini-batch of training examples. The mini-batch size is typically between 10 and 1,000. Mini-batch gradient descent is computationally less expensive than batch gradient descent because it uses a mini-batch of training examples to compute the gradient of the cost function. It is also less noisy than stochastic gradient descent because it uses more than one training example to compute the gradient of the cost function. Mini-batch gradient descent is the most common type of gradient descent used in practice.