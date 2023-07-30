# Decision Tree Ensemble
A decision tree ensemble is a collection of decision trees that are combined to make a prediction. The two most common methods for combining decision trees are bagging and boosting. Bagging is a method that involves training decision trees on different subsets of the training dataset. The predictions made by the trees are then combined. Boosting is a method that involves training decision trees in sequence, where each tree corrects the prediction errors made by the previous tree.

## Bagging
Bagging is an ensemble method that involves training the same algorithm many times using different subsets sampled from the training data. The final output prediction is averaged across the predictions of all of the sub-models. The three bagging models are:
- Bagged Decision Trees
- Random Forest
- Extra Trees

## Why Use Ensemble Methods?
The advantage of using an ensemble of trees is that it reduces the overall algorithm's sensitivity to what any single tree might be doing because each tree gets only one vote out of many. The robustness of the overall algorithm increases as it is less dependent on the decision of any single tree. The three main benefits of ensemble methods are:
- Improved Performance
- Improved Robustness
- Reduced Overfitting

## Sampling with Replacement
Sampling with replacement is a statistical technique where each selected item is returned back into the dataset before the next item is chosen. This method allows for the same item to be selected more than once, creating a diverse set of sample data. It's a fundamental technique used in bootstrapping, and more relevantly, in the creation of decision tree ensembles, such as random forests and boosted trees.

The reason we use sampling with replacement in decision tree ensembles is to ensure that each decision tree is trained on a slightly different set of data. This diversity in training data helps to create an array of decision trees that are not identical, but each carries unique characteristics. When these trees collectively vote on predictions, it brings forth a model that is less susceptible to overfitting, more generalizable, and more robust to small changes or noise in the data.