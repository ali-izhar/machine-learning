# Terminology

## Dataset
A dataset is a collection of data used to train and test a machine learning model. It is typically divided into two sections: the training set and the test set. Within the training set, the input variables are often referred to as inputs or features, and the output variables are known as targets.

A single input variable is represented by $x$, while a collection of input variables is symbolized by $X$. A singular target is represented by $y$, and a collection of targets is symbolized by $Y$. A single training example is expressed as $(x, y)$. A specific training example, referring to a particular row in the training set, is represented by $(x^{(i)}, y^{(i)})$, where $i$ denotes the index of the training example. The total number of training examples in the set is denoted by $m$.

When dealing with multiple features, each row represents a training example and each column represents a feature. The total number of features is denoted by $n$. For instance, the variable $x_j^{(i)}$ refers to the $j$th feature of the $i$th training example.

Here is a table representing a dataset with four training examples ($m=4$) and three features ($n=3$):

|         | Feature 1 | Feature 2 | Feature 3 |
|---------|-----------|-----------|-----------|
| Example 1 | $x_1^{(1)}$ | $x_2^{(1)}$ | $x_3^{(1)}$ |
| Example 2 | $x_1^{(2)}$ | $x_2^{(2)}$ | $x_3^{(2)}$ |
| Example 3 | $x_1^{(3)}$ | $x_2^{(3)}$ | $x_3^{(3)}$ |
| Example 4 | $x_1^{(4)}$ | $x_2^{(4)}$ | $x_3^{(4)}$ |


In this table, each row represents a training example, and each column is a feature. For example, $x_2^{(3)}$ denotes the second feature of the third training example.

- **Model:** A model refers to the mathematical construct that uses input data to generate predictions.

- **Parameters:** Parameters are the coefficients of the model. In the case of linear regression, the parameters are the slope $w$ and the intercept $b$.

- **Cost Function:** A cost function is a function that measures the performance of a model for a given dataset. It quantifies the error between predicted values and expected values and presents it in the form of a single real number. The goal of a machine learning model is to find the set of parameters that minimizes the cost function.

- **Gradient Descent:** Gradient descent is an optimization algorithm used to minimize the cost function. It works by iteratively updating the parameters in the direction of the negative gradient of the cost function. The gradient is the slope of the cost function, and it points in the direction of the greatest rate of increase of the cost function. Therefore, the negative gradient points in the direction of the greatest rate of decrease of the cost function, which is the direction we want to go in order to minimize the cost function.

- **Learning Rate:** The learning rate is a hyperparameter that controls how much the parameters of the model are adjusted at each step of gradient descent. A low learning rate will cause the model to take a long time to converge, while a high learning rate may cause the model to diverge.

- **Epoch:** An epoch is a single pass through the entire training set. In other words, one epoch is a single step of gradient descent.

- **Classes or Categories:** The goal of classification is to predict the category or class of a given set of data points. In case of binary classification, there are only two possible classes that are denoted as:

| Class 1 | Class 2 |
|---------|---------|
| 0       | 1       |
| false   | true    |
| no      | yes     |
| negative class| positive class|

In case of multi-class classification, there are more than two possible classes. For example, if we are trying to classify images of handwritten digits, the possible classes are the digits 0 through 9.

- **Decision Boundary:** A decision boundary is a hypersurface that separates the input space into two or more regions. In binary classification, the decision boundary is a curve that separates the input space into two regions, one for each class. In multi-class classification, the decision boundary is a hypersurface that separates the input space into multiple regions, one for each class.

- **Overfitting:** Overfitting occurs when a model performs well on the training set but poorly on the test set. This means that the model has learned the training data too well, and it is not able to generalize to new data. Overfitting can be caused by having too many parameters in the model, which allows the model to memorize the training data instead of learning the underlying patterns. It can also be caused by having too few training examples, which makes it difficult for the model to learn the underlying patterns. The term `overfitting` is also known as `high variance`.

- **Underfitting:** Underfitting occurs when a model performs poorly on both the training set and the test set. This means that the model is not able to learn the underlying patterns in the training data, and it is not able to generalize to new data. Underfitting can be caused by having too few parameters in the model, which makes it difficult for the model to learn the underlying patterns. It can also be caused by having too many training examples, which makes it difficult for the model to memorize the training data. The term `underfitting` is also known as `high bias`.

- **Regularization:** Regularization is a technique used to prevent overfitting. It does this by penalizing the cost function. The aim of regularization is:
- To reduce the mean squared error by minimizing the sum of the squared error.
- To encourage the model to choose smaller weights by adding a penalty to the cost function using the L1 or L2 norm of the weight vector.

The two most common regularization techniques are L1 and L2 regularization.

- **L1 Regularization:** L1 regularization adds a penalty equal to the sum of the absolute value of the coefficients. The L1 regularization penalty is computed as:

$$\lambda \sum_{j=1}^{n}|w_j|$$

where $\lambda$ is the regularization strength and $w_j$ is the weight for feature $j$.

- **L2 Regularization:** L2 regularization adds a penalty equal to the sum of the squared value of the coefficients. The L2 regularization penalty is computed as:

$$\lambda \sum_{j=1}^{n}w_j^2$$

where $\lambda$ is the regularization strength and $w_j$ is the weight for feature $j$.

- If $\lambda$ is too large, the model will be penalized heavily and the weights will be close to 0. This will result in underfitting as the model will be too simple (i.e. it will nearly equal to the term $b$).
- If $\lambda$ is too small, the model will not be penalized much and the weights will be large. This will result in overfitting as the model will be too complex.

- **L1 vs L2 Regularization:** L1 regularization is better than L2 regularization when the goal is to reduce the number of features. This is because L1 regularization tends to set the weights of irrelevant features to zero, thereby removing them from the model. L2 regularization is better than L1 regularization when the goal is to reduce the magnitude of the weights. This is because L2 regularization tends to shrink the weights of irrelevant features towards zero, but it does not set them to zero.

- **How does L2 Regularization work?** L2 regularization works by adding a penalty term to the cost function. The penalty term is equal to the sum of the squared value of the coefficients. The penalty term is multiplied by a regularization strength parameter $\lambda$ which controls how much the model is penalized. The regularization strength parameter $\lambda$ is a hyperparameter that needs to be tuned.

Recall that the cost function is given by:

$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})^2$$

where $h_w(x^{(i)})$ is the predicted value for the $i^{th}$ training example and $y^{(i)}$ is the actual value for the $i^{th}$ training example.

The cost function with L2 regularization is given by:

$$J(w) = \frac{1}{2m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n}w_j^2$$

where $\lambda$ is the regularization strength and $w_j$ is the weight for feature $j$. The regularization strength parameter $\lambda$ is a hyperparameter that needs to be tuned. By taking the derivative of the cost function with respect to the weights, we get:

$$\frac{\partial J(w)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m}w_j$$

When updating the weights, we subtract the derivative of the cost function with respect to the weights multiplied by the learning rate $\alpha$:

$$w_j := w_j - \alpha \frac{\partial J(w)}{\partial w_j}$$

Substituting the derivative of the cost function with respect to the weights into the equation above, we get:

$$w_j := w_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m}w_j \right)$$

Simplifying the equation above, we get:

$$w_j := w_j (1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})x_j^{(i)}$$

Notice that the term $w_j (1 - \alpha \frac{\lambda}{m})$ is the same as the term $w_j$ in the equation for gradient descent without regularization. The term $w_j (1 - \alpha \frac{\lambda}{m})$ is called the `weight decay term`. For example, if $\lambda = 0.1$ and $\alpha = 0.01$, then the weight decay term is equal to $0.99$. This means that the weights are multiplied by $0.99$ after each iteration. This causes the weights to decay towards zero, which helps prevent overfitting.