# Anomaly Detection: An Overview
Anomaly detection is an unsupervised learning algorithm that learns from an unlabeled dataset of normal events and detects unusual or anomalous events. This algorithm is used in various applications, such as fraud detection, manufacturing, and monitoring computer clusters.

## Density Estimation Technique
A common approach to anomaly detection is density estimation. It involves building a model for the probability of x (feature values), identifying regions of higher or lower probability. If the probability of a new test example (Xtest) is less than a small threshold (epsilon), it is flagged as an anomaly. If the probability is greater, it's considered normal.

## Gaussian Distribution
The Gaussian distribution is a continuous probability distribution that is symmetric about the mean. It is also known as the normal distribution. The Gaussian distribution is defined by the following probability density function:

$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi} \sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

where $\mu$ is the mean, $\sigma^2$ is the variance, and $\sigma$ is the standard deviation. The mean and variance can be estimated from the data using the following equations:

$$\mu = \frac{1}{m}\sum_{i=1}^{m}x^{(i)}$$

$$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(x^{(i)} - \mu)^2$$

## Anomaly Detection Algorithm
Given a training set of normal examples, $\{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$, where $x^{(i)} \in \mathbb{R}^n$. Each example $x^{(i)}$ is a vector of $n$ features.

The anomaly detection algorithm involves the following steps:

1. Choose $n$ features $x_1, x_2, ..., x_n$ that might be indicative of anomalous examples.
2. Fit parameters $\mu_1, \mu_2, ..., \mu_n, \sigma_1^2, \sigma_2^2, ..., \sigma_n^2$.
3. Given a new example $x$, compute $p(x)$:

$$p(x) = \prod_{j=1}^{n}p(x_j; \mu_j, \sigma_j^2) = \prod_{j=1}^{n}\frac{1}{\sqrt{2\pi} \sigma_j}e^{-\frac{(x_j-\mu_j)^2}{2\sigma_j^2}}$$

4. Anomaly if $p(x) < \epsilon$.

## Anomaly Detection vs. Supervised Learning
Anomaly detection is similar to supervised learning, but without the labels. In supervised learning, we have a training set of labeled examples, $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)})\}$, where $y^{(i)} \in \{0, 1\}$. The anomaly detection algorithm can be modified to use supervised learning by setting $y = 1$ if $p(x) < \epsilon$ and $y = 0$ if $p(x) \geq \epsilon$. The labeled examples can then be used to train a supervised learning algorithm.

| Anomaly Detection | Supervised Learning |
| --- | --- |
| Very small number of positive examples ($y = 1$) and large number of negative examples ($y = 0$). | Large number of positive and negative examples. |
| Many different types of anomalies. Hard for any algorithm to learn from positive examples what the anomalies look like. | Enough positive examples for algorithm to get a sense of what positive examples are like, future positive examples likely to be similar to ones in training set. |
| Future anomalies may look nothing like any of the anomalous examples we've seen so far. | Future positive examples likely to be similar to ones in training set. |
| Fraud detection, manufacturing, monitoring machines in a data center. | Email spam classification, weather prediction, cancer classification. |
| Can use Gaussian distribution to model $p(x)$. | Can use logistic regression or neural network to model $p(y|x)$. |

## Feature Selection and Transformation
The features used for anomaly detection should be indicative of anomalous examples. For example, if the features are not Gaussian or if the features are correlated, the Gaussian distribution will not model the data well. In this case, the features can be transformed to make them more Gaussian and less correlated.

### Tuning Features
The anomaly detection algorithm can be run with different features to see which ones work best. For example, if the features are not Gaussian, they can be transformed to make them more Gaussian. If the features are correlated, they can be combined to make them less correlated.

```python
# Experimenting with Transformations

# Plotting a histogram of feature X
plt.hist(x, bins=50)

# Transforming X with a square root
plt.hist(x**0.5, bins=50)

# Transforming X with a logarithm
plt.hist(np.log(x + 0.001), bins=50)
```

### Error Analysis for Anomaly Detection
The anomaly detection algorithm can be evaluated using a labeled cross-validation set. The algorithm can be run on the cross-validation set and the threshold $\epsilon$ can be varied to get a good value. The F1 score can be used to evaluate the algorithm.

Remember to apply the same transformation to your cross-validation and test set data. You can also carry out an error analysis process for anomaly detection, looking at where the algorithm is not doing well and coming up with improvements.

### Creating New Features
Sometimes it might be helpful to create new features by combining old features. For example, consider a video streaming data center that has two features: the throughput (mb/s) and latency (ms) of each server. A new feature, the ratio of throughput to latency, might be a significant feature in detecting anomalies in a data center's computer.