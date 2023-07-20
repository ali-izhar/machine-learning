# Neural Network Optimizers
Neural Network Optimizers are algorithms or methods used to change the attributes of the neural network such as weights and learning rate to reduce the losses. Optimizers help to get results faster.

## Gradient Descent
Gradient Descent is the most basic but fundamental optimization algorithm in machine learning and deep learning. It is an iterative optimization algorithm for finding the minimum of a function. The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the function at the current point, because this is the direction of steepest descent.

## Stochastic Gradient Descent (SGD)
Stochastic Gradient Descent (SGD) is a variation of the gradient descent algorithm that calculates the error and updates the model for each example in the training dataset, rather than the sum of the errors across all examples in the training dataset. The frequently updated model parameters result in a more robust model, albeit with a higher computational cost.

## Adagrad
Adaptive Gradient Algorithm (Adagrad) is an algorithm for gradient-based optimization which adapts the learning rate to the parameters, using low learning rates for parameters associated with frequently occurring features, and using high learning rates for parameters associated with infrequent features. Therefore, it is well-suited for dealing with sparse data.

## RMSprop
RMSprop (Root Mean Square Propagation) is an adaptive learning rate method proposed by Geoff Hinton. RMSprop divides the learning rate by an exponentially decaying average of squared gradients. Hinton suggests γ to be set to 0.9, while a good default value for the learning rate is 0.001.

## Adam
Adaptive Moment Estimation (Adam) is a method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients like RMSprop, Adam also keeps an exponentially decaying average of past gradients.

### Adam Algorithm Intuition
If $w_j$ (or $b$) keeps moving in the same direction, increase the learning rate. If $w_j$ (or $b$) keeps changing its direction (oscillating), reduce the learning rate.