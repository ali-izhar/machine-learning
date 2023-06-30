# Linear Regression
Linear regression is a supervised learning algorithm used to predict a continuous value. The goal of linear regression is to find the best fit line that describes the relationship between the independent and dependent variables. The best fit line is the one for which total prediction error (all data points) are as small as possible. Error is the distance between the point to the regression line.

## Model
A model is a function that maps inputs to outputs, and is denoted by $f$. The model takes as input a single training example, $(x^{(i)}, y^{(i)})$, and outputs a prediction, $\hat{y}^{(i)}$. The hat symbol above the $y$ indicates that the value is a prediction, and not the true value of $y$. For a linear regression model, the function $f$ is defined as follows:

$$\hat{y}^{(i)} = f(x^{(i)}) = w x^{(i)} + b$$

where $w$ is the weight, and $b$ is the bias. The weight and bias are the `parameters` of the model, and are used to make predictions. The goal of training a model is to find the optimal values for the parameters that minimize the error in the predictions (they are also sometimes referred to as `coefficients` or `weights`).

## Loss Function (Cost Function)
A loss function is a function that measures the error between the predicted value and the true value of the target. The loss function is denoted by $J$, and is a function of the parameters of the model. For a linear regression model, the loss function is defined as follows:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

This is called the `squared error cost function` or `mean squared error (MSE)`. The goal of training a model is to find the optimal values for the parameters that minimize the loss function. That is $\text{minimize } J(w, b)$.