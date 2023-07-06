# Terminology

## Dataset
A dataset is a collection of data used to train and test a machine learning model. It is typically divided into two sections: the training set and the test set. Within the training set, the input variables are often referred to as inputs or features, and the output variables are known as targets.

A single input variable is represented by $x$, while a collection of input variables is symbolized by $X$. A singular target is represented by $y$, and a collection of targets is symbolized by $Y$. A single training example is expressed as $(x, y)$. A specific training example, referring to a particular row in the training set, is represented by $(x^{(i)}, y^{(i)})$, where $i$ denotes the index of the training example. The total number of training examples in the set is denoted by $m$.

When dealing with multiple features, each row represents a training example and each column represents a feature. The total number of features is denoted by $n$. For instance, the variable $x_j^{(i)}$ refers to the $j$th feature of the $i$th training example.

Here is a table representing a dataset with four training examples ($m=4$) and three features ($n=3$):

\begin{tabular}{|c|c|c|}
\hline
$x_1^{(i)}$ & $x_2^{(i)}$ & $x_3^{(i)}$ \
\hline
$x_1^{(1)}$ & $x_2^{(1)}$ & $x_3^{(1)}$ \
$x_1^{(2)}$ & $x_2^{(2)}$ & $x_3^{(2)}$ \
$x_1^{(3)}$ & $x_2^{(3)}$ & $x_3^{(3)}$ \
$x_1^{(4)}$ & $x_2^{(4)}$ & $x_3^{(4)}$ \
\hline
\end{tabular}

In this table, each row represents a training example, and each column is a feature. For example, $x_2^{(3)}$ denotes the second feature of the third training example.

## Model
A model refers to the mathematical construct that uses input data to generate predictions. In the case of linear regression, the model is a function $f$ which predicts the target variable $(y)$ based on one or more predictor variables $(x)$.