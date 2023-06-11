# Machine Learning - Probability
This section focuses on the probability theory and its applications in machine learning.

## Random Variable
A random variable $X$ is a function that represents a random phenomenon, meaning its exact value cannot be determined. However, probabilities can be assigned to the possible values of $X$. For example, the random variable $X$ can represent the outcome of a coin toss, where $X$ can take on the values of heads or tails. The probability of $X$ being heads is $P(X = heads) = 0.5$ assuming a fair coin.

## Probability Distribution Function
A probability distribution function gives the probabilities of different outcomes for a random variable. The probability distribution can refer to either a Probability Mass Function (PMF) or a Probability Density Function (PDF). A PMF is used for discrete random variables, while a PDF is used for continuous random variables.

## Probability Mass Function (PMF)
A Probability Mass Function (PMF) gives the probabilities of different outcomes for a discrete random variable. A discrete random variable is a random variable that can only take on a countable number of values. For example, the number of heads obtained when flipping a coin three times (can be 0, 1, 2, or 3) is a discrete variable. The PMF of a fair coin toss is $P(X = heads) = 0.5$ and $P(X = tails) = 0.5$. The PMF of a fair six-sided die is $P(X = 1) = P(X = 2) = P(X = 3) = P(X = 4) = P(X = 5) = P(X = 6) = 1/6$.

## Probability Density Function (PDF)
A Probability Density Function (PDF) gives the probabilities of different outcomes for a continuous random variable. A continuous random variable is a random variable that can take on any value within a certain range, like temperature, height, weight, etc. It's important to note that for a continuous random variable, the PDF does not directly provide probabilities. Rather, it provides relative likelihoods. The probability of a specific outcome is technically zero because there is an infinite number of outcomes. The probability of an outcome within a certain range is the integral of the PDF over that range. 

## Cumulative Distribution Function (CDF)
The Cumulative Distribution Function (CDF) gives the probability that a random variable $X$ is less than or equal to a certain value $x$. The CDF is defined as $F(x) = P(X \leq x)$. In other words, it is the sum of probabilites of all outcomes less than or equal to $x$. 

> For continous random variables, the CDF is the integral of the PDF from negative infinity to $x$. For discrete random variables, the CDF is the sum of the PMF from negative infinity to $x$.

The value of the CDF always ranges from 0 to 1. The CDF is a monotonically increasing function, meaning it never decreases. The CDF is also a right-continuous function, meaning it has no jumps.

## Inverse Transform Sampling
Inverse Transform Sampling is a method for generating random numbers with a specified probability distribution.

> If $X$ is a random variable with CDF $F$, then $F(X)$ has a uniform distribution on $[0, 1]$. The uniform distribution here means that any value in the range $[0, 1]$ is equally likely to occur. This is due to the properties of the CDF, which by definition, ranges from 0 to 1 and is monotonically increasing.
>
> If you have a uniformly distributed random value $y$ from $[0, 1]$, you can transform this value into a value that follows any desired distribution using the inverse CDF of that distribution. The inverse CDF is the value $x$ such that $F(x) = y$. In essence, we're essentially "mapping back" from the uniform distribution on $[0, 1]$ to the original distribution of $X$.
>
> Therefore, if $Y$ is a random variable with a uniform distribution on $[0, 1]$, then $F^{-1}(Y)$ has the same distribution as $X$. This is the basis of inverse transform sampling.

## Generating Data
The process of generating data from a specified distribution is as follows:
1. Generate a uniformly distributed random value $y$ from $[0, 1]$ using a random number generator (RNG). This is implemented in the `generators.py` file as the `uniform_generator()` function.
2. Transform $y$ into a value $x$ that follows the desired distribution using the inverse CDF of that distribution. That is, $x = F^{-1}(y)$. In order to do this, we need to know the inverse CDF of the distribution. This is implemented in the `inverse.py` file for the gaussian and binomial distributions.

## Gaussian Distribution
The Gaussian distribution, also known as the normal distribution, is a continuous probability distribution that is symmetric about the mean. The Gaussian distribution is defined by the following probability density function:

$$PDF = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x - \mu}{\sigma})^2}$$

where $\mu$ is the mean and $\sigma$ is the standard deviation. The mean is the center of the distribution, while the standard deviation is the measure of the spread of the distribution. The Gaussian distribution is parameterized by the mean and standard deviation. If $X$ is a random variable with a Gaussian distribution, then $X \sim \mathcal{N}(\mu, \sigma)$. The CDF of the Gaussian distribution doesn't have a closed analytical expression, therefore, the closed formula uses a function called the Gaussian error function, denoted as $\text{erf}$. The CDF of the Gaussian distribution is defined as:

$$y = F(x) = \frac{1}{2} \left[ 1 + \text{erf} \left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right]$$

where $\text{erf}$ is the error function. The error function is defined as:

$$\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt$$

The inverse CDF of the Gaussian distribution is:

$$F^{-1}(y) = \mu + \sigma \sqrt{2} \text{erf}^{-1}(2y - 1)$$

> More on the Gaussian error function at [scipy.special.erf](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erf.html) and [scipy.special.erfinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfinv.html#scipy.special.erfinv). There is also a python implementation from the Python math library at [math.erf](https://docs.python.org/3/library/math.html#math.erf).