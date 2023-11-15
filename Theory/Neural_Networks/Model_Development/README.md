# Basic Recipe for Model Development
A basic recipe for model development is as follows:

## 1. High Bias (Underfitting)
First, we need to check if the model is doing well on the training set. If the model is not doing well on the training set (i.e. it is underfitting or has high bias), then we need to increase the model complexity. This can be done by:
    - Adding more features
    - Adding polynomial features
    - Decreasing the regularization parameter
    - Using a more complex or bigger model
    - Training the model longer
    - (NN architecture search)

## 2. High Variance (Overfitting)
If the model is doing well on the training set but not on the validation set (i.e. it is overfitting or has high variance), then we need to decrease the model complexity. This can be done by:
    - Adding more training data
    - Adding regularization
    - Using a simpler or smaller model
    - Reducing the training time
    - (NN architecture search)