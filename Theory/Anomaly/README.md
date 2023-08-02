# Anomaly Detection: An Overview
Anomaly detection is an unsupervised learning algorithm that learns from an unlabeled dataset of normal events and detects unusual or anomalous events. This algorithm is used in various applications, such as fraud detection, manufacturing, and monitoring computer clusters.

## Density Estimation Technique
A common approach to anomaly detection is density estimation. It involves building a model for the probability of x (feature values), identifying regions of higher or lower probability. If the probability of a new test example (Xtest) is less than a small threshold (epsilon), it is flagged as an anomaly. If the probability is greater, it's considered normal.

## Gaussian Distribution
The Gaussian distribution is a continuous probability distribution that is symmetric about the mean. It is also known as the normal distribution. The Gaussian distribution is defined by the following probability density function:

$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi} \sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

where $\mu$ is the mean and $\sigma^2$ is the variance.