# Feature Engineering
Feature engineering is an essential step in preparing data for machine learning. It's the process of creating new features or modifying existing ones to improve model performance. One key aspect of feature engineering is feature scaling. Feature scaling adjusts the range of the feature values to ensure that they are approximately on the same scale.

## Feature Scaling
Feature scaling is a method used to standardize the range of independent variables or features of data. In data processing, it's also known as data normalization. It's generally performed during the data preprocessing step.

### Why do we need Feature Scaling?
Many machine learning algorithms perform better or converge faster when features are on a relatively similar scale and/or close to normally distributed. Feature scaling is a crucial step in our data preprocessing pipeline because not only it helps to normalize the data, but it also speeds up the training process.

### Feature Scaling Methods
There are several methods to scale features. The most common ones are:

#### 1. Min-Max Scaling (Normalization)
<hr>
The simplest method is to rescale the range of features to [0, 1] or [-1,1]. This is called `min-max scaling` or `normalization`.

$$ X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}} $$

In this formula, $X$ is the original feature vector, $X_min$ is the minimum value of the feature, $X_max$ is the maximum value. The result is a new feature vector $X_norm$ which lies between 0 and 1 (inclusive).

#### 2. Mean Normalization
<hr>
Mean normalization is a more sophisticated method which not only scales the features to a similar range but also ensures that the distribution is centered around zero (mean = 0).

$$ X_{norm} = \frac{X - \mu}{X_{max} - X_{min}} $$

In this formula, $X$ is the original feature vector, $\mu$ is the mean of the feature vector.

#### 3. Z-Score Normalization (Standardization)
<hr>
Z-score normalization (also known as standardization) is a scaling technique that calculates the number of standard deviations away each point is from the mean. The result is that the features are rescaled to ensure the mean and the standard deviation to be 0 and 1, respectively.

$$ X_{norm} = \frac{X - \mu}{\sigma} $$

In this formula, $X$ is the original feature vector, $\mu$ is the mean of the feature vector, $\sigma$ is the standard deviation of the feature vector.
