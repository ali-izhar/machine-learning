# Linear Regression :leaves:
Linear regression is a supervised learning algorithm for predicting a continuous target variable based on one or more predictor variables. It aims to find a best-fit line (or hyperplane in multiple regression) capturing the relationship between independent (predictor) and dependent (target) variables.

- [Simple Linear Regression](#simple-linear-regression)
- [Multiple Linear Regression](#multiple-linear-regression)

## Simple Linear Regression
Simple linear regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables:

$$Y = \beta_0 + \beta_1X + \epsilon$$

where:
- $\beta_0$ is the intercept
- $\beta_1$ is the slope
- $\epsilon$ is the error term

The error term $\epsilon$ is assumed to have the following properties:
- $\epsilon$ is a random variable that is normally distributed
- $E(\epsilon) = 0$
- $Var(\epsilon) = \sigma^2$
- $Cov(\epsilon_i, \epsilon_j) = 0$ for all $i \neq j$

The goal of linear regression is to minimize the vertical distance between all the data points and the fitted line. The vertical distance between a data point and the fitted line is called a residual.

$$e_i = y_i - \hat{y}_i$$

The goal is to minimize the sum of squared residuals (SSR).

$$SSR = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

The least squares estimates of the regression coefficients $\beta_0$ and $\beta_1$ are the values that minimize the SSR.

$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = r_{xy}\frac{S_y}{S_x}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$$

## Partitioning the Variance
The total sum of squares (SST) is divided into two parts: the regression sum of squares (SSR) and the residual sum of squares (SSE). The regression sum of squares measures the amount of variation in the response that is explained by the regression model. The residual sum of squares measures the amount of variation in the response that is not explained by the regression model.

| Source | DF | SS | MS | F |
| --- | --- | --- | --- | --- |
| Regression | 1 | SSR | MSR = SSR | MSR/MSE |
| Residual | n - 2 | SSE | MSE = SSE/(n - 2) | |
| Total | n - 1 | SST | | |

**Question:** Now that we've fitted a regression line, do the assumptions of linear regression hold?
- $\epsilon$ is a random variable that is normally distributed. To check this assumption, we can plot a histogram or a QQ plot of the residuals. If the histogram or QQ plot is approximately normal, then this assumption is satisfied.
- $E(\epsilon) = 0$. To check this assumption, we can calculate the mean of the residuals. If the mean is approximately 0, then this assumption is satisfied. Alternatively, we can plot the residuals against the fitted values. If the residuals are centered around 0, then this assumption is satisfied.
- $Var(\epsilon) = \sigma^2$. To check this assumption, we can calculate the variance of the residuals. If the variance is approximately constant, then this assumption is satisfied. Alternatively, we can plot the residuals against the fitted values. If the residuals are spread equally along the horizontal axis, then this assumption is satisfied.

**Question:** How important is $x$ in predicting $y$?
**Answer:** We can use the $t$-test to test the null hypothesis $H_0: \beta_1 = 0$.

$$H_0: \beta_1 = 0 \quad H_a: \beta_1 \neq 0$$

The test statistic is:

$$t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)}$$

where:

$$SE(\hat{\beta}_1) = \frac{S_{\epsilon}}{\sqrt{S_{xx}}} = \frac{\sqrt{SSE/(n - 2)}}{\sqrt{S_{xx}}}$$

The $p$-value is:

$$p = P(|T| > |t|) = 2P(T > |t|)$$

where $T$ is a $t$-distribution with $n - 2$ degrees of freedom.

**Question:** How well does the model fit the data?
**Answer:** Assess the goodness of fit using the coefficient of determination $R^2$.

The coefficient of determination $R^2$ is the proportion of the variance in the response variable that is explained by the regression model:

$$R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$

**Question:** What is the relationship between $R^2$ and the correlation coefficient $r$?
**Answer:** $R^2 = r^2$

## Performing Linear Regression with Normal Equation
The `normal equation` is a closed-form solution for finding the optimal parameters of a linear regression model. It is given by:

$$W = (X^T X)^{-1} X^T y$$

Here, $X$ is the feature matrix, $y$ is the target vector, and $W$ is the weight vector. The normal equation can be derived by setting the gradient of the loss function $J(W)$ to zero and solving for $W$.

The normal equation is computationally efficient for small datasets, but it is not suitable for large datasets because the matrix $X^T X$ is a square matrix of size $n \times n$, where $n$ is the number of features. The computational complexity of inverting such a matrix is $O(n^3)$. For large datasets, we can use gradient descent to find the optimal parameters.

## Multiple Linear Regression
Multiple linear regression is a statistical method that allows us to summarize and study relationships between two or more continuous (quantitative) variables:

$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_pX_p + \epsilon$$

where:
- $\beta_0$ is the intercept
- $\beta_1, \beta_2, \ldots, \beta_p$ are the slopes
- $\epsilon$ is the error term

The error term $\epsilon$ is assumed to have the following properties:
- $\epsilon$ is a random variable that is normally distributed
- $E(\epsilon) = 0$
- $Var(\epsilon) = \sigma^2$

The goal of multiple linear regression is to minimize the sum of squared residuals (SSR).

$$SSR = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

The least squares estimates of the regression coefficients $\beta_0, \beta_1, \ldots, \beta_p$ are the values that minimize the SSR.

$$\hat{\beta}_j = \frac{\sum_{i=1}^n (x_{ij} - \bar{x}_j)(y_i - \bar{y})}{\sum_{i=1}^n (x_{ij} - \bar{x}_j)^2} = \frac{S_{xy_j}}{S_{xx_j}} = r_{xy_j}\frac{S_y}{S_{x_j}}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}_1 - \hat{\beta}_2\bar{x}_2 - \cdots - \hat{\beta}_p\bar{x}_p$$

**Question:** Are all the predictors useful for predicting $y$?
**Answer:** We can use the $F$-test to test the null hypothesis $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$.

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0 \quad H_a: \text{at least one } \beta_j \neq 0$$

The test statistic is:

$$F = \frac{(SST - SSE)/p}{SSE/(n - p - 1)}$$

where:
- $SST = \sum_{i=1}^n (y_i - \bar{y})^2$
- $SSE = \sum_{i=1}^n (y_i - \hat{y}_i)^2$

The $p$-value is:

$$p = P(F > f)$$

where $F$ is an $F$-distribution with $p$ and $n - p - 1$ degrees of freedom.

**Question:** If the hypothesis testing results in rejecting $H_0$, then which predictors are useful for predicting $y$?

### Backward Elimination (p-value approach)
1. Select a significance level $\alpha$ to stay in the model (e.g. $\alpha = 0.05$).
2. Fit the full model with all possible predictors.
3. Consider the predictor with the highest $p$-value. If $p > \alpha$, go to Step 4. Otherwise, go to Step 5.
4. Remove the predictor.
5. Fit the model without this predictor. Go to Step 3.