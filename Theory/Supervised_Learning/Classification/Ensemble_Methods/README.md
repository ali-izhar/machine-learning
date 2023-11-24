# Ensemble Methods
An ensemble method is a powerful technique that allows us to combine multiple machine learning models into one model, which often performs better than any of the individual models alone.

## Why Creating an Ensemble is a Good Idea?
Assume that we have an ensemble of $n$ classifiers, each of which has an error rate of $\epsilon$. We also assume that the ensemble uses a majority voting on the classifiers' predictions to make its final prediction.

Let's try to compute the error rate of the ensemble in this case. First, we denote the number of classifiers that made a wrong prediction by $k$ (out of $n$ classifiers). 

If the base classifiers are independent (i.e., their errors are uncorrelated), then the variable $k$ follows a binomial distribution with parameters $n$ and $\epsilon$, that is, $k \sim Bin(n, \epsilon)$, since each prediction is a Bernoulli trial with probability $\epsilon$ of success. The ensemble will make a wrong prediction only if at least half of the base classifiers are wrong (since it uses a majority voting). Therefore, according to the Binomial probability distribution, the error rate of the ensemble is:

$$\epsilon_{ensemble} = P(k \geq \frac{n}{2}) = \sum_{i=\frac{n}{2}}^{n} \binom{n}{i} \epsilon^i (1-\epsilon)^{n-i}$$

For example, if we have 25 base classifiers, each with an error rate of 0.25, then the error rate of the ensemble would be:

$$\epsilon_{ensemble} = \sum_{i=13}^{25} \binom{25}{i} 0.25^i 0.75^{25-i} = 0.0034$$

The error rate of the ensemble is much lower than the error rate of the base classifiers (0.25). This is because the base classifiers are independent, and the ensemble makes a wrong prediction only if at least half of the base classifiers are wrong.

We can actually plot a graph of the ensemble error rate as a function of the base error rate $\epsilon$ (assuming that we have 25 base classifiers):

<div style="align="center>
    <img src="media/ensemble_error.png" width="500">
</div>

We can see that the turning point is at $\epsilon = 0.5$, which means that if the base classifiers are no better than random guessing, then the ensemble will not perform better than random guessing. However, if the base classifiers perform better than random guessing, then the ensemble will perform better than the base classifiers.

From the above discussion we can learn that there are two conditions under which the ensemble performs better than the individual classifiers:

1. The base classifiers should be independent of each other. In practice, it is difficult to ensure total independence between the classifiers. However, practice has shown that even when the classifiers are partially correlated, the ensemble can still perform better than any one of them.
2. Each base classifier should have an error rate of less than 0.5, i.e., it should perform better than a random guesser. A model that performs only slightly better than a random guesser is called a weak learner. As we have just shown, an ensemble of many weak learners can become a strong learner.

## Voting
A simple way to aggregate the predictions of the base models in the ensemble is by using voting. Each base model makes a prediction and votes for each sample. Then the ensemble returns the class with the highest votes.

1. Majority Voting: The ensemble selects the class with the highest number of votes.
2. Weighted Voting: Each base model is assigned a weight, and the ensemble selects the class with the highest sum of weights.
3. Soft Voting: The ensemble selects the class with the highest average predicted probability.