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

## Vector Notation
For convenience and computational efficiency, we can express the multiple linear regression model using vector notation. We define $W=[w_1, w_2, ..., w_n]$ as the row vector of weights and $X=[x_1^{(i)}, x_2^{(i)}, ..., x_n^{(i)}]$ as the row vector of features for a given sample. The model then becomes:

$$\hat{y}^{(i)} = f(x^{(i)}) = W X^{(i)T} + b$$

Note that $X^{(i)T}$ is the transpose of the feature vector, resulting in a dot product operation with the weight vector. In this form, the training process aims to minimize the loss function $J(W, b)$ to find the optimal parameter values.