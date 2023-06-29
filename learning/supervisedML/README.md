# Terminology

## Dataset
A dataset refers to the collection of data used to train and test a machine learning model. It is typically divided into two sections: the training set and the test set. Within the training set, the input variables are often referred to as `inputs` or `features`. A single input variable is represented by `$x$`, while a collection of input variables is symbolized by `$X$`.

The output variables in the training set are known as `targets`. A singular target is represented by `$y$`, and a collection of targets is symbolized by `$Y$`.

In this context, a single training example is expressed as `$(x, y)$`. A specific training example, referring to a particular row in the training set, is represented by `$(x^{(i)}, y^{(i)})$`, where $i$ denotes the index of the training example. The total number of training examples in the set is denoted by `$m$`.

## Model
A model is a function that maps inputs to outputs, and is denoted by `$f$`. The model takes as input a single training example, $(x^{(i)}, y^{(i)})$, and outputs a prediction, `$\hat{y}^{(i)}$`. The hat symbol above the $y$ indicates that the value is a prediction, and not the true value of $y$. For a linear regression model, the function $f$ is defined as follows:

$$\hat{y}^{(i)} = f(x^{(i)}) = w x^{(i)} + b$$

where $w$ is the weight, and $b$ is the bias. The weight and bias are the `parameters` of the model, and are used to make predictions. The goal of training a model is to find the optimal values for the parameters that minimize the error in the predictions (they are also sometimes referred to as `coefficients` or `weights`).

## Loss Function (Cost Function)
A loss function is a function that measures the error between the predicted value and the true value of the target. The loss function is denoted by `$J$`, and is a function of the parameters of the model. For a linear regression model, the loss function is defined as follows:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

This is called the `squared error cost function` or `mean squared error (MSE)`. The goal of training a model is to find the optimal values for the parameters that minimize the loss function.