# Classification
Classification algorithms are used for predicting the category of given data. They are used in various real-world applications like spam detection, image recognition, and disease prediction, among others. In binary classification problems, the model predicts one of two possible classes, while in multiclass classification, the model can predict more than two classes.

## Separation and Classification of 2 Populations
**Goal:** Given features from observations $X= \{x_1, x_2, ..., x_n\}$, we want to classify them into two populations $C_1$ and $C_2$.

A `cost matrix` is used to quantify the cost of misclassification. 

|  | $C_1$ | $C_2$ |
| --- | --- | --- |
| $C_1$ | 0 | $C_{12}$ |
| $C_2$ | $C_{21}$ | 0 |

Where $C_{12}$ is the cost of misclassifying a $C_1$ observation as $C_2$ and $C_{21}$ is the cost of misclassifying a $C_2$ observation as $C_1$. 

The expected cost of misclassification is given by:

$$ECM = C_{12}P(x \rightarrow C_1 \wedge x \in C_2) + C_{21}P(x \rightarrow C_2 \wedge x \in C_1)$$

Read as: The expected cost of misclassification is the cost of misclassifying a $C_1$ observation as $C_2$ times the probability of misclassifying a $C_1$ observation as $C_2$ plus the cost of misclassifying a $C_2$ observation as $C_1$ times the probability of misclassifying a $C_2$ observation as $C_1$.

The goal is to minimize the expected cost of misclassification.

Let $R_1$ be the region where $x \rightarrow C_1$ and $R_2$ be the region where $x \rightarrow C_2$.
- $R_1 \cap R_2 = \emptyset$
- $R_1 \cup R_2 = X$

$$p(x \rightarrow C_1 \wedge x \in C_2) = p(x \in R_1 \wedge x \in C_2) = p(x \in R_1 \mid x \in C_2)p(x \in C_2) = \int_{R_1} f_2(x)dx \cdot P(C_2)$$

$$p(x \rightarrow C_2 \wedge x \in C_1) = p(x \in R_2 \wedge x \in C_1) = p(x \in R_2 \mid x \in C_1)p(x \in C_1) = \int_{R_2} f_1(x)dx \cdot P(C_1)$$

Therefore, the expected cost of misclassification is given by:

$$ECM = C_{12} \int_{R_1} f_2(x)dx \cdot P(C_2) + C_{21} \int_{R_2} f_1(x)dx \cdot P(C_1)$$

Since,

$$\int_{R_1} f_1(x)dx + \int_{R_2} f_1(x)dx = 1$$

$$ECM = C_{12} \int_{R_1} f_2(x)dx \cdot P(C_2) + C_{21} (1 - \int_{R_1} f_1(x)dx) \cdot P(C_1)$$

$$ECM = C_{12} \int_{R_1} f_2(x)dx \cdot P(C_2) + C_{21} \cdot P(C_1) - C_{21} \int_{R_1} f_1(x)dx \cdot P(C_1)$$

$$ECM = C_{12} \times P(C_2) \times \int_{R_1} f_2(x)dx + C_{21} \times P(C_1) \times \int_{R_2} f_1(x)dx$$

$$ECM = C_{12} \times P(C_2) \times \int_{R_1} f_2(x)dx + C_{21} \times P(C_1) \times (1 - \int_{R_1} f_1(x)dx)$$

$$ECM = C_{12} \times P(C_2) \times \int_{R_1} f_2(x)dx + C_{21} \times P(C_1) - C_{21} \times P(C_1) \times \int_{R_1} f_1(x)dx$$