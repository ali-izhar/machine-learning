# Classification
Classification is a supervised learning approach in which the computer program learns from the data input given to it and then uses this learning to classify new observation.

## Types of Classification
- Linear Models
- Support Vector Machines
- Decision Trees
- Random Forest
- Naive Bayes
- K-Nearest Neighbors
- Neural Networks

## Linear Models
Linear models are the simplest classification models. They are called linear because a linear function (a line in 2D, a plane in 3D) is used to discriminate between the two classes. The most common linear classification algorithm is the logistic regression.

## Sigmoind Function (Logistic Function)
The sigmoid function is a mathematical function used to map the predicted values to probabilities. It maps any real value into another value between 0 and 1. In machine learning, we use sigmoid to map predictions to probabilities.

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

## Logistic Regression
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).

$$ f_{w,b}(x) = w^Tx + b$$
$$ g(z) = \sigma(w^Tx + b) = \frac{1}{1+e^{-(w^Tx + b)}}$$