# Classification
Classification algorithms are used for predicting the category of given data. They are used in various real-world applications like spam detection, image recognition, and disease prediction, among others. In binary classification problems, the model predicts one of two possible classes, while in multiclass classification, the model can predict more than two classes.

## Separation and Classification of 2 Populations
**Goal:** Given features from observations $X= \{x_1, x_2, ..., x_n\}$, we want to classify them into two populations $\pi_{1}$ and $\pi_{2}$.

A `cost matrix` is used to quantify the cost of misclassification. 

|  | $\pi_{1}$ | $\pi_{2}$ |
| --- | --- | --- |
| $\pi_{1}$ | 0 | $c(1 \mid 2)$ |
| $\pi_{2}$ | $c(2 \mid 1)$ | 0 |

Where $c(1 | 2)$ is the cost of misclassifying a $\pi_{1}$ observation as $\pi_{2}$ and $c(2 | 1)$ is the cost of misclassifying a $\pi_{2}$ observation as $\pi_{1}$.

The expected cost of misclassification is given by:

$$ECM = c(1 | 2) P(x \rightarrow \pi_{1} \wedge x \in \pi_{2}) + c(2 | 1) P(x \rightarrow \pi_{2} \wedge x \in \pi_{1})$$

<i>Read as: The expected cost of misclassification is the cost of misclassifying a $\pi_{1}$ observation as $\pi_{2}$ times the probability of misclassifying a $\pi_{1}$ observation as $\pi_{2}$ plus the cost of misclassifying a $\pi_{2}$ observation as $\pi_{1}$ times the probability of misclassifying a $\pi_{2}$ observation as $\pi_{1}$</i>

The goal is to minimize the expected cost of misclassification.

Let $R_1$ be the region where $x \rightarrow \pi_{1}$ and $R_2$ be the region where $x \rightarrow \pi_{2}$.
- $R_1 \cap R_2 = \emptyset$
- $R_1 \cup R_2 = X$

Let $p_1$ be the probability of $x \in \pi_{1}$ and $p_2$ be the probability of $x \in \pi_{2}$. Let $f_1(x)$ be the probability density function of $x$ given $x \in \pi_{1}$ and $f_2(x)$ be the probability density function of $x$ given $x \in \pi_{2}$. Then:

$$p(x \rightarrow \pi_1 \wedge x \in \pi_2) = p(x \in R_1 \wedge x \in \pi_2) = p(x \in R_1 | x \in \pi_2)p(x \in \pi_2) = \int_{R_1} f_2(x) dx \times p_2$$

$$p(x \rightarrow \pi_1 \wedge x \in \pi_2) = p(x \in R_1 \wedge x \in \pi_2) = p(x \in R_1 | x \in \pi_2)p(x \in \pi_2) = \int_{R_1} f_2(x) dx \times p_2$$

$$p(x \rightarrow \pi_2 \wedge x \in \pi_1) = p(x \in R_2 \wedge x \in \pi_1) = p(x \in R_2 | x \in \pi_1)p(x \in \pi_1) = \int_{R_2} f_1(x) dx \times p_1$$

$$\text{Therefore,}$$

$$\text{ECM} = c(1 | 2) \int_{R_1} f_2(x) dx \times p_2 + c(2 | 1) \int_{R_2} f_1(x) dx \times p_1$$

$$\text{Since,}$$

$$\int_{R_1} f_1(x) dx + \int_{R_2} f_1(x) dx = 1$$

$$\text{ECM} = c(1 | 2) \times p_2 \int_{R_1} f_2(x) dx + c(2 | 1) \times p_1 \left[ 1 - \int_{R_1} f_1(x) dx \right]$$

$$\text{ECM} = c(1 | 2) \times p_2 \int_{R_1} f_2(x) dx + c(2 | 1)p_1 - c(2 | 1)p_1 \int_{R_1} f_1(x) dx$$

$$\text{ECM} = c(2 | 1)p_1 + c(1 | 2)p_2 \int_{R_1} f_2(x) dx - c(2 | 1)p_1 \int_{R_1} f_1(x) dx$$

$$\text{ECM} = c(2 | 1)p_1 + \left[ c(1 | 2)p_2f_2(x) - c(2 | 1)p_1f_1(x) \right] dx$$

$$\text{To minimize ECM, we want the integral term to be less than or equal to zero.}$$

$$c(1 | 2)p_2f_2(x) - c(2 | 1)p_1f_1(x) \leq 0$$

$$c(1 | 2)p_2f_2(x) \leq c(2 | 1)p_1f_1(x)$$

$$\frac{c(1 | 2) p_2}{c(2 | 1) p_1} \leq \frac{f_1(x)}{f_2(x)}$$

Where $\frac{c(1 | 2)}{c(2 | 1)}$ is the `cost ratio`, $\frac{p_1}{p_2}$ is the `prior ratio`, and $\frac{f_1(x)}{f_2(x)}$ is the `density ratio`.

The left-hand side of the inequality is a threshold value. If the density ratio is greater than the threshold value, then $x \in \pi_1$. If the density ratio is less than the threshold value, then $x \in \pi_2$.
