# Polynomial Regression
One of the most common forms of non-linear regression is polynomial regression, where the relationship between the features and the label is modeled by a polynomial of some degree. The degree of the polynomial is usually a hyperparameter of the model, unless we have some prior knowledge about the relationship between the variables.

Recall that in regression problems we are given a set of $n$ labeled examples: $D=\{(x_1,y_1),\ldots,(x_n,y_n)\}$, where $x_i$ is an $m$-dimensional vector containing the features of the $i$-th example and $y_i$ is the label of the $i$-th example.

In polynomial regression, we model the target variable $y$ as a polynomial of some degree $d$ of the input $x$, i.e., our model hypothesis is:

$$h(x) = w_0 + w_1 x + w_2 x^2 + \ldots + w_d x^d$$

The key observation here is that we can treat the powers of $x$: $x, x^2, \ldots, x^d$ as distinct independent variables. Then, polynomial regression becomes a special case of multiple linear regression, since the model is still linear in the parameters that need to be estimated.

Therefore, we can find the optimal parameters $w*$ using the same techniques that we used to solve multiple linear regression problems, namely, the closed-form solution or gradient descent.