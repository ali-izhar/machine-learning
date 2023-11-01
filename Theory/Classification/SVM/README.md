# Support Vector Machines (SVM)

- **Support Vectors**: Data points crucial for SVM decision-making.
- **Hyperplane and Margin**: SVM draws boundaries using hyperplanes. The goal is to maximize the space (margin) between data groups.
- **Maximum Margin Classifier**: Finds the boundary with the largest gap between classes, ensuring the nearest points from each class are maximally separated.
- **Soft Margin Classifier**: Accounts for imperfect data. Allows some points to be on the wrong side of the margin or boundary, improving handling of noisy data.

## SVM Kernels
A kernel is a function that measures the similarity between two data points in a transformed feature space. The kernel function allows us to perform non-linear transformations of the original data into a higher-dimensional space where it may be easier to separate the classes in a classification problem or model the relationships in a regression problem.

**Linear Kernel**: The linear kernel is the simplest kernel function and is used when the data is linearly separable. It is defined as the dot product between the input features.
$$K(x,y)=x^T.y$$
For example, if we have two data points $x=[2,3]$ and $y=[1,4]$, then the linear kernel is calculated as follows:
$$K(x,y)=x^T.y=(2,3).(1,4)=2.1+3.4=2+12=14$$

<hr>

**Polynomial Kernel**: The polynomial kernel is used when the data has a polynomial structure. It is defined as the dot product between the input features raised to a power.
$$K(x,y)=(x^T.y+c)^d$$
For example, if we have two data points $x=[2,3]$ and $y=[1,4]$, then the polynomial kernel is calculated as follows:
$$K(x,y)=(x^T.y+c)^d=((2,3).(1,4)+1)^2=(2+12+1)^2=15^2=225$$

<hr>

**Radial Basis Function (RBF) Kernel**: The RBF kernel is the most commonly used kernel function and is used when the data is not linearly separable. It is defined as a Gaussian function of the Euclidean distance between the input features.
$$K(x,y)=e^(\frac{-||x-y||^2}{2\sigma^2})$$

<hr>

**Sigmoid Kernel**: The sigmoid kernel is used when the data has a sigmoidal structure. It is defined as a hyperbolic tangent function of the dot product between the input features.
$$K(x,y)=tanh(\alpha.x^T.y+c)$$

<hr>

## Hinge Loss
Hinge loss is a loss function used for training classifiers. The hinge loss is used for "maximum-margin" classification, most notably for support vector machines (SVMs). For an intended output $y=1$ and a classifier score $f(x)$, the hinge loss of the prediction $y$ is defined as:

$$\ell(y)=max(0,1-y.f(x))$$

- If the original output is $y=1$ and the model output $f(x)$ is equal to or greater than one, the hinge loss will be zero.
- If the original output is $y=1$ and the model output $f(x)$ is less than one, the hinge loss will be non-zero and proportional to the distance from one.
- The hinge loss is always non-negative and convex - thus its easy to optimize using gradient descent methods.

## SVM Loss
The SVM loss is a hinge loss plus a regularization term. The regularization term is used to penalize large weights and is defined as the squared L2 norm of the weight vector:

The hinge loss for a single data point $x_k$ and its true lable $y_k$ is defined as:

$$
\ell\_k=\sum_{l\neq y\_k} \max(0,f(x\_k)\_l-f(x\_k)\_{y\_k}+\Delta)
$$


- $f(x_k)_l$ is the score for the $l$-th class of the $k$-th data point.
- $f(x_k)_{y_k}$ is the score for the true class of the $k$-th data point.
- $\Delta$ is the margin parameter.

The total SVM loss is defined as the average of the hinge loss for all data points plus the regularization term:

$$L(W)=\frac{1}{N}\sum_{k=1}^N\ell_k+\lambda\sum_{i,j}W_{i,j}^2$$

- $N$ is the number of data points.
- $\lambda$ is the regularization strength.

## SVM Optimization
The SVM optimization problem is to minimize the SVM loss function $L(W)$ with respect to the weight matrix $W$. The SVM loss function is convex, so the global minimum is guaranteed to be found. The SVM loss function is also differentiable, so gradient descent can be used to find the global minimum.
