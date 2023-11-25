# Regression Problems
In regression problems, we are given a set of $n$ labeled examples: $D={(x_1,y_1),...,(x_n,y_n)}$ where each $x_i$ is a vector that consists of $m$ features: $x_i=(x_{i1},...,x_{im})^t$. The variables $x_{ij}$ are called the independent variables or the explanatory variables. The label $y$ is a continuous-valued variable $(y \in \mathbb{R})$ and is called the dependent variable or the response variable. 

We assume that there is a correlation between the label $y$ and the input vector $x$, which is modeled by some function $f(x)$ and an error variable $\epsilon$:

$$y=f(x)+\epsilon$$

Our goal is to find the function $f(x)$, since knowing this function will allow us to predict the labels for any new sample. However, since we have a limited number of training samples from which to learn the function $f(x)$, we can only approximate it. The function that our model estimates from the given data is called the model's hypothesis and is typically denoted by $h(x)$.

## Linear Regression
In linear regression, we assume that there is a linear relationship between the features and the target label. Therefore, the model's hypothesis takes the following form:

$$h(x)=w_0+w_1x_1+...+w_mx_m$$

Where $w_0, w_1,...,w_m$ are the parameters of the model. The parameter $w_0$ is often called the intercept (or bias), since it represents the intersection point of the graph of $h(x)$ with the y-axis (in two dimensions). 

To simplify $h(x)$, we add a constant feature $x_0$ that is always equal to 1. This allows us to write $h(x)$ as the dot product between the feature vector $x=(x_0,x_1,...,x_m)^t$ and the parameter vector $w=(w_0,w_1,...,w_m)^t$:

$$h(x)= W^tX = \sum_{j=0}^{m}w_jx_j$$

## Ordinary Least Squares (OLS)
Our goal in linear regression is to find the parameters $w$ that will make our model's predictions $h(x)$ be as close as possible to the true labels $y$. In other words, we would like to find the model's parameters that best fit the data set. To that end, we define a **cost function** (sometimes also called an error function) that measures how far our model's predictions are from the true labels.

We start by defining the residual as the difference between the label of a given data point and the value predicted by the model:

$$r_i=y_i-h(x_i)$$

**Ordinary least squares (OLS)** regression finds the optimal parameter values that minimize the sum of squared residuals:

$$J(w)=\sum_{i=1}^{n}r_i^2=\sum_{i=1}^{n}(y_i-h(x_i))^2=\sum_{i=1}^{n}(y_i-W^tX_i)^2$$

Note that a loss function calculates the error per observation and in OLS it is called the squared loss, while a cost function (typically denoted by $J$) calculates the error over the whole data set, and in OLS it is called the sum of squared residuals (SSR) or sum of squared errors (SSE).

## Simple Linear Regression
When the data set has only one feature (i.e., when it consists of two-dimensional points $(x_i,y_i)$), the regression problem is called simple linear regression. Geometrically, in simple linear regression, we are trying to find a straight line that goes as close as possible through all the data points:

<div style="align: center">
    <img src="media/simple_linear_regression.png" width="500">
</div>

In this case, the model’s hypothesis is simply the equation of the line:

$$h(x)=w_0+w_1x$$

Where $w_1$ is the slope of the line and $w_0$ is the intercept. The residuals in this case are the vertical distances between the data points and the line. The least squares cost function is the sum of the squared residuals:

$$J(w_0, w_1)=\sum_{i=1}^{n}(y_i-h(x_i))^2=\sum_{i=1}^{n}(y_i-(w_0+w_1x_i))^2$$

## The Normal Equations
Our objective is to find the parameters $w_0$ and $w_1$ of the line that best fits the points, i.e., the line that leads to the minimum cost. To that end, we can take the partial derivatives of $J(w_0, w_1)$ with respect to $w_0$ and $w_1$, set them to zero, and then solve the resulting system of equations (which are called the normal equations).

$$
\begin{align*}
\frac{\partial J(w_0, w_1)}{\partial w_0} &= \frac{\partial}{\partial w_0} \sum_{i=1}^{n} (y_i - (w_0 + w_1x_i))^2 & \text{(definition of $J$)} \\
&= \sum_{i=1}^{n} \frac{\partial}{\partial w_0}(y_i - (w_0 + w_1x_i))^2 & \text{(sum of derivatives)} \\
&= \sum_{i=1}^{n} 2(y_i - (w_0 + w_1x_i))\frac{\partial}{\partial w_0}(y_i - (w_0 + w_1x_i)) & \text{(chain rule of derivatives)} \\
&= \sum_{i=1}^{n} 2(y_i - (w_0 + w_1x_i)) \cdot (-1) & \text{(partial derivative)} \\
&= \sum_{i=1}^{n} 2(w_0 + w_1x_i - y_i)
\end{align*}
$$

Setting this derivative to 0 yields the following:

$$
\begin{align*}
\sum_{i=1}^{n} 2(w_0 + w_1x_i - y_i) &= 0 \\
n\cdot w_0 + w_1\sum_{i=1}^{n} x_i - \sum_{i=1}^{n} y_i &= 0 \\
w_0 = \frac{\sum_{i=1}^{n} y_i - w_1\sum_{i=1}^{n} x_i}{n}
\end{align*}
$$

Similarly, we can take the partial derivative of $J(w_0, w_1)$ with respect to $w_1$:

$$
\begin{align*}
\frac{\partial J(w_0, w_1)}{\partial w_1} &= \frac{\partial}{\partial w_1} \sum_{i=1}^{n} (y_i - (w_0 + w_1x_i))^2 & \text{(definition of $J$)} \\
&= \sum_{i=1}^{n} \frac{\partial}{\partial w_1}(y_i - (w_0 + w_1x_i))^2 & \text{(sum of derivatives)} \\
&= \sum_{i=1}^{n} 2(y_i - (w_0 + w_1x_i))\frac{\partial}{\partial w_1}(y_i - (w_0 + w_1x_i)) & \text{(chain rule of derivatives)} \\
&= \sum_{i=1}^{n} 2(y_i - (w_0 + w_1x_i)) \cdot (x_i) & \text{(partial derivative)} \\
\end{align*}
$$

Setting this derivative to 0 yields the following:

$$
\begin{align*}
\sum_{i=1}^{n} 2(y_i - (w_0 + w_1x_i)) \cdot (x_i) &= 0 \\
\sum_{i=1}^{n} x_iy_i - w_0\sum_{i=1}^{n} x_i - w_1\sum_{i=1}^{n} x_i^2 &= 0 \\
\end{align*}
$$

Let's substitute $w_0$ in the second equation with the value we found for it in the first equation:

$$
\begin{align*}
\sum_{i=1}^{n} x_iy_i - \left(\frac{\sum_{i=1}^{n} y_i - w_1\sum_{i=1}^{n} x_i}{n}\right)\sum_{i=1}^{n} x_i - w_1\sum_{i=1}^{n} x_i^2 &= 0 \\
\sum_{i=1}^{n} x_iy_i - \left(\frac{\sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i}{n}\right) + w_1 \left(\frac{(\sum_{i=1}^{n} x_i)^2}{n}\right) - w_1\sum_{i=1}^{n} x_i^2 &= 0 \\
w_1 \left[\left(\sum_{i=1}^{n} x_i\right)^2 - n\sum_{i=1}^{n} x_i^2\right] &= n\sum_{i=1}^{n} x_iy_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i \\
w_1 &= \frac{n\sum_{i=1}^{n} x_iy_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i}{n\sum_{i=1}^{n} x_i^2 - (\sum_{i=1}^{n} x_i)^2}
\end{align*}
$$

Therefore, the optimal values for $w_0$ and $w_1$ are:

$$
w_0 = \frac{\sum_{i=1}^{n} y_i - w_1\sum_{i=1}^{n} x_i}{n}
$$

$$
w_1 = \frac{n\sum_{i=1}^{n} x_iy_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i}{n\sum_{i=1}^{n} x_i^2 - (\sum_{i=1}^{n} x_i)^2}
$$

## Evaluation Metrics
There are several evaluation metrics that are used to evaluate the performance of regression models. The two most common ones are RMSE (Root Mean Squared Error) and $R^2$ (R-squared) score.

Note the difference between an evaluation metric and a cost function. A cost function is used to define the objective of the model’s learning process and is computed on the training set. Conversely, an evaluation metric is used after the training process to evaluate the model on a holdout data set (a validation or a test set).

### Root Mean Squared Error (RMSE)
RMSE is defined as the square root of the mean of the squared errors (the differences between the model’s predictions and the true labels):

$$RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-h(x_i))^2}$$

Note that what we called residuals during the model’s training are typically called errors (or prediction errors) when they are computed over the holdout set.

RMSE is always non-negative, and a lower RMSE means the model has a better fit to the data (a perfect model has an RMSE of 0).

### R-squared ($R^2$) Score
The $R^2$ score (also called the coefficient of determination) is a measure of the goodness of fit of a model. It computes the ratio between the sum of squared errors of the regression model and the sum of squared errors of a baseline model that always predicts the mean value of $y$, and subtracts the result from 1:

$$R^2=1-\frac{\sum_{i=1}^{n}(y_i-h(x_i))^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}$$

Where $\bar{y}$ is the mean value of $y$:

$$\bar{y}=\frac{1}{n}\sum_{i=1}^{n}y_i$$

The $R^2$ score is always between 0 and 1, and a higher $R^2$ score means the model has a better fit to the data (a perfect model has an $R^2$ score of 1).
