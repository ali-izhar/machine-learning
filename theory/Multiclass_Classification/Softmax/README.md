# Softmax Regression (Multinomial Logistic Regression)
Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes. In logistic regression we assumed that the labels were binary: $y(i) \in \{0,1\}$. We used such a classifier to distinguish between two kinds of hand-written digits. Softmax regression allows us to handle $y(i) \in \{1,\ldots,K\}$ where $K$ is the number of classes.

## Hypothesis Representation
We now introduce a way to generalize logistic regression to classification problems where the class label $y$ can take on more than two possible values. We consider a multiclass classification problem where the label $y$ can be $1, 2, \ldots, K$.

In logistic regression, we had the following hypothesis function:

$$z = w \dot x + b$$

$$a_1 = \sigma(z) = \frac{1}{1 + e^{-z}} = P(y = 1 | x)$$

$$a_2 = 1 - a_1 = \frac{e^{-z}}{1 + e^{-z}} = P(y = 0 | x)$$

$$L(a_1, a_2) = -y \log(a_1) - (1 - y) \log(1 - a_1) = -y \log(a_1) - (1 - y) \log(a_2)$$

$$L(a_1, a_2) = \cases{-\log(a_1) & if $y = 1$ \cr -\log(a_2) & if $y = 0$}$$

For a multiclass classification, the softmax function $\sigma : \mathbb{R}^K \rightarrow \mathbb{R}^K$ is defined as follows:

$$a_1 = \sigma(z)_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + \ldots + e^{z_K}} = P(y = 1 | x)$$

$$a_2 = \sigma(z)_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2} + \ldots + e^{z_K}} = P(y = 2 | x)$$

$$\vdots$$

$$a_K = \sigma(z)_K = \frac{e^{z_K}}{e^{z_1} + e^{z_2} + \ldots + e^{z_K}} = P(y = K | x)$$

$$L(a_1, a_2, \ldots, a_K, y) = \cases{-\log(a_1) & if $y = 1$ \cr -\log(a_2) & if $y = 2$ \cr \vdots & \vdots \cr -\log(a_K) & if $y = K$}$$

Notice that $\text{loss} = -\log(a_i) \text{ if } y = i$. If we plot the graph of $-\log(x)$, we can see that the loss is high when $x$ is close to $0$ and low when $x$ is close to $1$. This is called the **cross-entropy loss**.

## Cost Function
The cost function for softmax regression is defined as follows:

$$J(w, b) = \frac{1}{m} \sum_{i=1}^m L(a_1^{(i)}, a_2^{(i)}, \ldots, a_K^{(i)}, y^{(i)})$$

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K 1\{y^{(i)} = k\} \log \frac{e^{z_k^{(i)}}}{\sum_{j=1}^K e^{z_j^{(i)}}}$$

In tensorflow, we can use the `SparseCategoricalCrossentropy` function to compute the cost function. The name `SparseCategoricalCrossentropy` means that the labels are integers instead of one-hot vectors (i.e. $y \in \{1, 2, \ldots, K\}$ instead of $y \in \{0, 1\}^K$).

> The word "sparse" in `SparseCategoricalCrossentropy` means that the labels are integers instead of one-hot vectors. The word "categorical" means that we are doing multiclass classification.

## Tensorflow Implementation
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(loss=SparseCategoricalCrossentropy())
model.fit(X_train, y_train, epochs=10)
```

This implementation is correct. However, it is not efficient and can result in numerical instability due to round-off errors. For instance, consider computing $\frac{2}{10000} = 1 + \frac{1}{10000} - (1 - \frac{1}{10000})$ in two different ways up to 18 decimal places:

```python
x1 = 2.0 / 10000
print(f"{x1:.18f}")
# 0.000200000000000000

x2 = 1.0 + 1.0 / 10000 - (1.0 - 1.0 / 10000)
print(f"{x2:.18f}")
# 0.000199999999999978
```

The first way is more accurate than the second way. The second way is more prone to round-off errors. The second way is also more computationally expensive because it requires more operations.

## Efficient Implementation
The loss function of a logistic regression is defined as follows:

$$L(a, y) = -y \log(a) - (1 - y) \log(1 - a)$$

where $a = \sigma(z) = \frac{1}{1 + e^{-z}}$. In this case, the round-off error does not have a significant impact on the loss function. A more accurate loss (in code) is given as:

$$L(a, y) = -y \log(\frac{1}{1 + e^{-z}}) - (1 - y) \log(1 - \frac{1}{1 + e^{-z}})$$

By avoiding the middle step of computing $a = \sigma(z)$ before computing the loss, we can avoid round-off errors.

Since the softmax function has $K$ outputs between $0$ and $1$ that sum to $1$, the round-off error can have a significant impact on the loss function. A more accurate loss (in code) is given as:

$$L(a_1, a_2, \ldots, a_K, y) = -\sum_{k=1}^K 1\{y = k\} \log \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

```python
model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(10, activation='linear')
])

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
model.fit(X, Y, epochs=10)
logit = model(X)
f_x = tf.nn.sigmoid(logit)
```

Since we're not computing the middle step of computing $a = \sigma(z)$, we're not applying any activation function to the last layer. Therefore, we need to use `activation='linear'` instead of `activation='softmax'` and also set `from_logits=True` in `SparseCategoricalCrossentropy`.