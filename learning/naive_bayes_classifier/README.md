# Generating synthetic data utilizing the generator functions
We will utilize the generator functions (see [here](../../learning/probability/generators.py)) to generate synthetic data for the naive bayes classifier. The dataset will contain information about three different dog breeds. Once we have prepared the dataset, we will train the naive bayes classifier to classify the dog breeds based on the features of the dogs.

## Generating the dataset
The dataset will contain information about three different dog breeds. The features of the dogs will be the following:
- height (in cm), which follows a normal distribution
- weight (in kg), which follows a normal distribution
- bark_days, representing the number of days (out of 30) that the dog barked, which follows a binomial distribution with $n=30$.
- ear_head_ratio, which is the ratio between the length of the ears and the length of the head, which follows a uniform distribution.

$$FEATURES = ['height', 'weight', 'bark_days', 'ear_head_ratio']$$

