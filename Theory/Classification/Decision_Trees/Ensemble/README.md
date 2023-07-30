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

The reason we use sampling with replacement in decision tree ensembles is to ensure that each decision tree is trained on a slightly different set of data. This diversity in training data helps to create an array of decision trees that are not identical, but each carries unique characteristics. When these trees collectively vote on predictions, it brings forth a model that is less susceptible to overfitting, more generalizable, and more robust to small changes or noise in the data.,

## Generating a Tree Ensemble
The first step in generating a tree ensemble is to create a bootstrap sample of the training data. This is done by randomly selecting rows from the training data with replacement. The number of rows selected is equal to the number of rows in the training data. This bootstrap sample is then used to train a decision tree. This process is repeated for each decision tree in the ensemble. The final step is to combine the predictions made by each tree in the ensemble. The most common method for combining the predictions is to use a majority vote for classification problems and an average for regression problems.

## Bagged Decision Trees
```python
# the choice of B is arbitrary, usually between 64 and 100.
for b = 1 to B:
    # create a bootstrap sample of the training data
    bootstrap_sample = train.sample(n=len(train), replace=True)
    # train a decision tree on the bootstrap sample
    tree = DecisionTreeClassifier()
    tree.fit(bootstrap_sample)
```

### Further randomization of bagged decision trees
At each node, when choosing a feature to use to split, if $n$ features are available, pick a random subset of $k \leq n$ features and allow the algorithm to only choose from that subset of features. If n is large, then k is usually set to $\sqrt{n}$.

## XGBoost
XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. XGBoost stands for `Extreme Gradient Boosting.` It is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks. However, when it comes to small-to-medium structured/tabular data, decision tree based algorithms are considered best-in-class right now. Please note that XGBoost is not an actual tree ensemble method, but it is a gradient boosting library that also uses decision trees as its base learners.
```python
for b = 1 to B:
    # create a bootstrap sample of the training data
    bootstrap_sample = train.sample(n=len(train), replace=True)
    # instead of picking from all examples with (1/m) probability,
    # make it more likely to pick examples with high error or pick
    # misclassified examples from previous trained trees
    tree = DecisionTreeClassifier()
    tree.fit("new dataset")
```

## When to use Decision Tree Ensembles
| Decision Tree Ensembles | Neural Networks |
| --- | --- |
| Works well on tabular (structured) data | Works well on all types of data, including tabular (structured) and unstructured data (images, text, audio, etc.) |
| Works well on small-to-medium sized data | Works well on large data |
| Fast to train | May be slower than decision tree ensembles to train |
| Small decision trees may be human interpretable | Not human interpretable |
| - | Works well with transfer learning |
| - | When building a system of multiple models working together, it is easier to integrate neural networks |