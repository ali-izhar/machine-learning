# Evaluating Machine Learning Models

## Introduction
Systematically evaluating a machine learning model's performance is crucial for understanding how well it generalizes to new data.

## Data Splitting
The first step is to split your data into two parts:

- **Training set** - Fit the model on this subset of the data.
- **Test set** - Evaluate the model on this subset of the data.

```python
# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def train_test_split(X, y, test_size=0.3):
    m = len(y)
    test_set_size = int(m * test_size)
    shuffled_indices = np.random.permutation(m)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
```

## Regression Model Evaluation
For regression models, where the task is predicting a continuous target value, two important metrics are:

- **Training error** - Average error on training data. Lower values indicate better fit.
- **Test error** - Average error on held-out test data. Lower values indicate better generalization.

For example, with a squared error cost function:

```python
# average training error
J_train = (1/2*m) * sum((y_train - y_train_pred) ** 2)

# average test error
J_test = (1/2*m) * sum((y_test - y_test_pred) ** 2)
```

A large gap between training and test error indicates overfitting.

## Classification Model Evaluation
For classification models, where the task is predicting a discrete class label, two useful metrics are:

- **Training accuracy** - Fraction of correct predictions on training data. Higher is better.
- **Test accuracy** - Fraction of correct predictions on held-out test data. Higher is better.

```python
# fraction correct on training set
accuracy_train = sum(y_train == y_train_pred) / m_train

# fraction correct on test set
accuracy_test = sum(y_test == y_test_pred) / m_test
```

Again, a large gap between training and test accuracy indicates overfitting.

## Conclusion
By evaluating models on held-out test data, you can identify overfitting and select the best model for your problem. The techniques here form the basis for more advanced methods like `cross-validation`.