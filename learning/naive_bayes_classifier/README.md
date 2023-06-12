# Generating synthetic data utilizing the generator functions
We will utilize the [generator functions](./probability/generators.py) to generate synthetic data for the naive bayes classifier. The dataset will contain information about three different dog breeds. Once we have prepared the dataset, we will train the naive bayes classifier to classify the dog breeds based on the features of the dogs.

## Generating the dataset
The dataset will contain information about three different dog breeds. The features of the dogs will be the following:
- height (in cm), which follows a normal distribution
- weight (in kg), which follows a normal distribution
- bark_days, representing the number of days (out of 30) that the dog barked, which follows a binomial distribution with $n=30$.
- ear_head_ratio, which is the ratio between the length of the ears and the length of the head, which follows a uniform distribution.

`FEATURES = ['height', 'weight', 'bark_days', 'ear_head_ratio']`

## Naive Bayes Algorithm
Let $X$ be a set of training data. An element $x \in X$ is a tuple of the form $(x_1, x_2, ..., x_n)$, where $x_i$ is the value of the $i$-th feature. For instance, in our example, $x = (height, weight, bark\_days, ear\_head\_ratio)$. Let $C$ be a set of classes that we want to classify the elements of $X$ to. In our example, $C$ is the set of dog breeds.

Suppose there are $m$ classes $C_1, C_2, ..., C_m$. Suppose there are $m=5$ different types of dog breeds in the training data. The idea is to predict the class of a sample $x \in X$ by calculating the probability of $x$ belonging to each class $C_i$ and then choosing the class with the highest probability.

$$\text{predicted class for } x = \arg \max \limits_{i=1,2,...,m} P(C_i \mid x)$$

So, if the highest value of $P(C_i \mid x)$ is $P(C_3 \mid x)$, then the predicted class for $x$ is $C_3$.

## Calculating the probability of a class
The probability of a class $C_i$ is calculated as follows:

$$P(C_i \mid x) = \frac{P(x \mid C_i) P(C_i)}{P(x)}$$

Note that $P(x)$ is the same for all classes (it is positive and constant) for every class $C_i$, therefore, to maximize $P(C_i \mid x)$, we only need to maximize $P(x \mid C_i) P(C_i)$. The term $P(x \mid C_i)$ is called the `likelihood` and the term $P(C_i)$ is called the `class prior` probability and it denotes how likely a random sample from $X$ (without knowing any of its features) belongs to the class $C_i$. This value is usually not known and can be estimated by the frequency of the class in the training data. However, if the training set is too small, it is common to assume that each class is equally likely to occur, i.e. $P(C_i) = \frac{1}{m}$ for all $i=1,2,...,m$, thus only maximizing $P(x \mid C_i)$ remains.

In general, it would be computationally expensive to calculate $P(x \mid C_i)$ for all $i=1,2,...,m$ and then choose the class with the highest probability. However, the naive bayes algorithm makes the assumption of `class-conditional independence`. This assumption states that each attribute is independent of each other attribute with each class. In other words, the value of one feature does not depend on the value of another feature. This assumption allows us to calculate $P(x \mid C_i)$ as follows:

$$P(x \mid C_i) = P(x_1, x_2, ..., x_n \mid C_i) = \prod_{j=1}^{n} P(x_j \mid C_i)$$

The probabilities $\mathbf P(x_k \mid C_i)$ can be estimated from the training data. The computation of $\mathbf P(x_k \mid C_i)$ depends on whether $x_k$ is categorical or not.

- If $x_k$ is categorical, then $\mathbf P(x_k \mid C_i)$ is the number of samples in $X$ that have attribute $x_k$ divided by the number of samples in class $C_i$.

- If $x_k$ is continuous-valued or discrete-valued, we need to make an assumption about its distribution and estimate its parameters using the training data. For instance, if $x_k$ is continuous-valued, we can assume that $\mathbf P(x_k \mid C_i)$ follows a Gaussian distribution with parameters $\mu_{C_i}$ and $\sigma_{C_i}$. Therefore, we need to estimate $\mu$ and $\sigma$ from the training data, and then

$$\mathbf P(x_k \mid C_i) = \text PDF_{gaussian} (x_k, \mu_{C_i}, \sigma_{C_i})$$
