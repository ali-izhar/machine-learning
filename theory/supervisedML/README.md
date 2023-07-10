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

## Regression
- **Model:** A model refers to the mathematical construct that uses input data to generate predictions. In the case of linear regression, the model is a function $f$ which predicts the target variable $(y)$ based on one or more predictor variables $(x)$.

- **Parameters:** Parameters are the coefficients of the model. In the case of linear regression, the parameters are the slope $w$ and the intercept $b$.

- **Cost Function:** A cost function is a function that measures the performance of a model for a given dataset. It quantifies the error between predicted values and expected values and presents it in the form of a single real number. The goal of a machine learning model is to find the set of parameters that minimizes the cost function.

- **Gradient Descent:** Gradient descent is an optimization algorithm used to minimize the cost function. It works by iteratively updating the parameters in the direction of the negative gradient of the cost function. The gradient is the slope of the cost function, and it points in the direction of the greatest rate of increase of the cost function. Therefore, the negative gradient points in the direction of the greatest rate of decrease of the cost function, which is the direction we want to go in order to minimize the cost function.

- **Learning Rate:** The learning rate is a hyperparameter that controls how much the parameters of the model are adjusted at each step of gradient descent. A low learning rate will cause the model to take a long time to converge, while a high learning rate may cause the model to diverge.

- **Epoch:** An epoch is a single pass through the entire training set. In other words, one epoch is a single step of gradient descent.

## Classification
The goal of classification is to predict the category or class of a given set of data points. In case of binary classification, there are only two possible classes that are denoted as:
| Class 1 | Class 2 |
|---------|---------|
| 0       | 1       |
| false   | true    |
| no      | yes     |
| negative class| positive class|

In case of multi-class classification, there are more than two possible classes. For example, if we are trying to classify images of handwritten digits, the possible classes are the digits 0 through 9.

- **Decision Boundary:** A decision boundary is a hypersurface that separates the input space into two or more regions. In binary classification, the decision boundary is a curve that separates the input space into two regions, one for each class. In multi-class classification, the decision boundary is a hypersurface that separates the input space into multiple regions, one for each class.

