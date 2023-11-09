# Linear Regression
Linear regression is a supervised learning algorithm used for predicting a continuous target variable based on one or more predictor variables. The goal is to find the best fit line that captures the relationship between the independent (predictor) and dependent (target) variables.

Linear regression can be categorized into two types based on the number of predictor variables used: simple linear regression (one predictor) and multiple linear regression (more than one predictor).

## Simple Linear Regression
Given an independent variable $X$ and dependent variable $Y$ such that we have reasons to believe that there exists a linear relationship between $X$ and $Y$, then the linear model is:

$$Y = \beta_0 + \beta_1 X + \epsilon$$

where $\beta_0$ is the intercept, $\beta_1$ is the slope, and $\epsilon$ is the error term. The goal of linear regression is to find the best fit line that minimizes the sum of squared errors (SSE) between the actual and predicted values of $Y$.

## Simple Linear Regression Objective Function
In linear regression, the objective function is the sum of squared errors (SSE) between the actual and predicted values of $Y$:

$$\text{minimize } \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where $y_i$ is the actual value of $Y$ and $\hat{y}_i$ is the predicted value of $Y$. The error term $\epsilon$ is assumed to be normally distributed with mean 0 and variance $\sigma^2$. To minimize the SSE, we need to find the optimal values for the intercept $\beta_0$ and slope $\beta_1$.

$$\text{minimize } \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2$$

$$b_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{Cov(x, y)}{Var(x)} = r_{xy} \frac{s_y}{s_x}$$

$$b_0 = \bar{y} - b_1 \bar{x}$$

where $\bar{x}$ is the mean of $X$ and $\bar{y}$ is the mean of $Y$. The slope $b_1$ is the change in $Y$ divided by the change in $X$. The intercept $b_0$ is the mean of $Y$ minus the slope times the mean of $X$.

## Partitioning Variability
The total variability in $Y$ can be partitioned into two components: the variability explained by the regression line and the unexplained variability. The total sum of squares (SST) is the sum of the squared differences between the actual values of $Y$ and the mean of $Y$:

$$SST = \sum_{i=1}^{n} (y_i - \bar{y})^2$$

The regression sum of squares (SSR) is the sum of the squared differences between the predicted values of $Y$ and the mean of $Y$:

$$SSR = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2$$

The error sum of squares (SSE) is the sum of the squared differences between the actual and predicted values of $Y$:

$$SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

The total sum of squares can be partitioned into the regression sum of squares and the error sum of squares:

$$SST = SSR + SSE$$

The coefficient of determination $R^2$ is the proportion of the total variability in $Y$ that is explained by the regression line:

$$R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$

The coefficient of determination $R^2$ is a measure of the strength of the linear relationship between $X$ and $Y$. It is the square of the correlation coefficient $r$ between $X$ and $Y$:

$$R^2 = r^2$$

## ANOVA Table
The ANOVA table is a table that summarizes the results of the analysis of variance. It is used to test the significance of the regression line. The ANOVA table is as follows:

| Source of Variation | Sum of Squares | Degrees of Freedom | Mean Square | F-Statistic |
| :--- | :--- | :--- | :--- | :--- |
| Regression | SSR | $k$ | $MSR = \frac{SSR}{k}$ | $F = \frac{MSR}{MSE}$ |
| Error | SSE | $n - k - 1$ | $MSE = \frac{SSE}{n - k - 1}$ | |
| Total | SST | $n - 1$ | | |

where $k$ is the number of predictor variables and $n$ is the number of observations. The F-statistic is the ratio of the mean square for regression to the mean square for error. The null hypothesis $H_0$ is that the regression line does not explain a significant amount of the variability in $Y$. The alternative hypothesis $H_a$ is that the regression line explains a significant amount of the variability in $Y$. If the F-statistic is greater than the critical value, then we reject the null hypothesis and conclude that the regression line explains a significant amount of the variability in $Y$. Otherwise, we fail to reject the null hypothesis and conclude that the regression line does not explain a significant amount of the variability in $Y$.

## Multiple Linear Regression
Given $k$ independent variables $X_1, X_2, ..., X_k$ and a dependent variable $Y$ such that we have reasons to believe that there exists a linear relationship between $X_1, X_2, ..., X_k$ and $Y$, then the linear model is:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_k X_k + \epsilon$$

where $\beta_0$ is the intercept, $\beta_1, \beta_2, ..., \beta_k$ are the slopes, and $\epsilon$ is the error term. The goal of linear regression is to find the best fit hyperplane that minimizes the sum of squared errors (SSE) between the actual and predicted values of $Y$. The hyperplane is a $k$-dimensional plane in a $k$-dimensional space.

## Multiple Linear Regression Objective Function
In multiple linear regression, the objective function is the sum of squared errors (SSE) between the actual and predicted values of $Y$:

$$\text{minimize } \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where $y_i$ is the actual value of $Y$ and $\hat{y}_i$ is the predicted value of $Y$. The error term $\epsilon$ is assumed to be normally distributed with mean 0 and variance $\sigma^2$. To minimize the SSE, we need to find the optimal values for the intercept $\beta_0$ and slopes $\beta_1, \beta_2, ..., \beta_k$. The optimal values can be found using the normal equation or gradient descent.

Question: Are all of the variables 

## Multiple Linear Regression Normal Equation
The normal equation is a closed-form solution for finding the optimal parameters of a linear regression model. It is given by:

$$\beta = (X^T X)^{-1} X^T y$$

Here, $X$ is the feature matrix, $y$ is the target vector, and $\beta$ is the weight vector. The normal equation can be derived by setting the gradient of the loss function $J(\beta)$ to zero and solving for $\beta$.

The normal equation is computationally efficient for small datasets, but it is not suitable for large datasets because the matrix $X^T X$ is a square matrix of size $k \times k$, where $k$ is the number of features. The computational complexity of inverting such a matrix is $O(k^3)$. For large datasets, we can use gradient descent to find the optimal parameters.

## Multiple Linear Regression Gradient Descent
Gradient descent is an optimization algorithm used in machine learning and deep learning models to minimize a function iteratively. It's frequently used to find the optimal solution to many problems.

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