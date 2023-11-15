# Learning Curves - Understanding Your Model's Performance
Learning curves are a powerful tool for understanding the performance of a learning algorithm with respect to the amount of experience it has (for instance, the number of training examples).

## Training Error and Cross-Validation Error
As the training set size increases, we observe that the cross-validation error generally decreases. This makes sense as with more data, the algorithm learns a better model.

Contrary to this, the training error increases as the training set size increases. This is because it becomes increasingly difficult for a quadratic function to perfectly fit all training examples.

It is important to note that the cross-validation error is typically higher than the training error since the parameters are fit to the training set. Hence, the model is expected to perform better on the training set than on the cross-validation set.

## High Bias vs High Variance
Learning curves can also shed light on whether a model suffers from high bias (underfitting) or high variance (overfitting).

A high bias scenario is seen when a simple linear function is fitted. The training and cross-validation errors both tend to flatten out after a while. This is because the model doesn't change much more with the addition of more examples, hence the errors plateau. **If your learning algorithm has high bias, collecting more training data won't significantly improve the performance.** In this case, you should focus on improving the model (using a more complex model, decreasing regularization, etc.).

In a high variance scenario, the training error increases gradually with the increase in training set size and the cross-validation error is significantly higher than the training error. This indicates that the model performs much better on the training set than on the cross-validation set. **In this case, increasing the training set size could help a lot as it reduces the cross-validation error and improves the performance of the algorithm.**