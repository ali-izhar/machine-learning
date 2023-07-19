# Logistic Regression
Logistic regression is a statistical technique capable of predicting a binary outcome. It's a classification algorithm that is used for the prediction of the probability of a categorical dependent variable. The dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.).

## Concept
In its essence, logistic regression models the probability that an input (or set of inputs) belongs to a particular class. The logistic function, also called the `sigmoid function`, is used for this purpose, allowing the model to output a value between 0 and 1, which can be interpreted as a probability.

The formula for logistic regression hypothesis is:

$$f_{w,b}(x) = \sigma(w^Tx + b)$$

where:
- $f_{w,b}(x)$ is the hypothesis
- $\sigma$ is the sigmoid function
- $w$ is the weight vector
- $b$ is the bias
- $x$ is the input vector of features

The sigmoid function can map any value to a value from 0 to 1, making it useful for models that need to predict probabilities.

The sigmoid function is defined as:

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

## Cost Function
In contrast to linear regression, which uses mean squared error as the cost function, logistic regression uses a `logarithmic loss function`, also known as the cost function. This is due to the fact that in logistic regression, the decision boundary is not a straight line and the output values are not continuous but lie between 0 and 1.

The loss function for a single training example is defined as:

$$L(f_{w,b}(x^{(i)}), y^{(i)}) = \begin{cases} 
-log(f_{w,b}(x^{(i)})) & \text{if } y^{(i)} = 1 \\ 
-log(1-f_{w,b}(x^{(i)})) & \text{if } y^{(i)} = 0 
\end{cases}$$

A more compact way to write the loss function is:

$$J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}log(f_{w,b}(x^{(i)})) + (1-y^{(i)})log(1-f_{w,b}(x^{(i)}))$$

This is because when $y^{(i)}=1$, the second term $(1-y^{(i)})log(1-f_{w,b}(x^{(i)}))$ becomes 0. Similarly, when $y^{(i)}=0$, the first term $y^{(i)}log(f_{w,b}(x^{(i)}))$ becomes 0.

The cost function for the entire training set is the average of the loss function for each training example:

$$J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}log(f_{w,b}(x^{(i)})) + (1-y^{(i)})log(1-f_{w,b}(x^{(i)}))$$

## Cost Function Intuition
The negative log function is a convex function shown below. This means that gradient descent will always converge to the global minimum.

- **Case 1:** $y=1$
The graph of $-log(f_{w,b}(x^{(i)}))$ is shown below:
<img src="media/negative_log1.png" width=250px>

In this case, if $f_{w,b}(x) = 1$, the loss is 0. If $f_{w,b}(x) \rightarrow 0$, the loss goes to infinity. Therefore, the model will try to predict a probability close to 1 for $y=1$ and penalize the model heavily if it predicts a probability close to 0.

- **Case 2:** $y=0$
The graph of $-log(1-f_{w,b}(x^{(i)}))$ is shown below:
<img src="media/negative_log2.png" width=250px>

In this case, if $f_{w,b}(x) = 0$, the loss is 0. If $f_{w,b}(x) \rightarrow 1$, the loss goes to infinity. Therefore, the model will try to predict a probability close to 0 for $y=0$ and penalize the model heavily if it predicts a probability close to 1.

## Gradient Descent
Gradient Descent is an iterative optimization algorithm used to minimize a function, in this case, the cost function. The algorithm repeatedly takes steps proportional to the negative of the gradient of the function at the current point to reach a local or global minimum.

The update rules in the context of logistic regression are:

$$w := w - \alpha \frac{\partial J(w,b)}{\partial w}$$

$$b := b - \alpha \frac{\partial J(w,b)}{\partial b}$$

where:
- $w, b$ are the parameters of the model that we will optimize
- $\alpha$ is the learning rate
- $J(w,b)$ is the cost function

## Logistic Regression with Regularization
Regularization is a technique used to prevent overfitting by discouraging overly complex models in some way. In the case of logistic regression, L2 regularization is often used, which involves adding an extra term to the cost function.

The cost function for logistic regression with L2 regularization is:

$$J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}log(f_{w,b}(x^{(i)})) + (1-y^{(i)})log(1-f_{w,b}(x^{(i)})) + \frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2$$

where:
- $\lambda$ is the regularization parameter
- $n$ is the number of features