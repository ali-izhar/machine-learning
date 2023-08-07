# Recommender Systems
Recommender systems are a collection of algorithms used to recommend items to users based on information taken from the user. These systems have become ubiquitous and can be commonly seen in online stores, movies databases and job finders.

## Movie Rating Example
Let's say we have a movie database with 5 movies and 4 users. Each user has rated some of the movies on a scale of 1 to 5. The question mark '?' indicates that the user has not rated the movie yet (maybe the user has not watched the movie yet). Let's denote the number of users by $n_u$ and the number of movies by $n_m$. Let's set $r(i,j) = 1$ if user $j$ has rated movie $i$ and $r(i,j) = 0$ otherwise. Let's set $y^{(i,j)}$ equal to the rating given by user $j$ to movie $i$ if $r(i,j) = 1$. The following table shows the ratings of the movies by the users.

$$n_u = 4, n_m = 5, n (features) = 2$$

| Movie | User 1 | User 2 | User 3 | User 4 |
|-------|--------|--------|--------|--------|
| M1    | 5      | 5      | 0      | 0      |
| M2    | 5      | ?      | ?      | 0      |
| M3    | ?      | 4      | 0      | ?      |
| M4    | 0      | 0      | 5      | 4      |
| M5    | 0      | 0      | 5      | ?      |

## Adding Features to the Movie Rating Example
Now, let's say we have some information about the movies.

| Movie | User 1 | User 2 | User 3 | User 4 | x1 (Romance) | x2 (Action) |
|-------|--------|--------|--------|--------|--------------|-------------|
| M1    | 5      | 5      | 0      | 0      | 0.9          | 0           |
| M2    | 5      | ?      | ?      | 0      | 1.0          | 0.01        |
| M3    | ?      | 4      | 0      | ?      | 0.99         | 0           |
| M4    | 0      | 0      | 5      | 4      | 0.1          | 1.0         |
| M5    | 0      | 0      | 5      | ?      | 0            | 0.9         |

The features x1 and x2 are the genres of the movies. The values of the features are between 0 and 1. For user 1, predict rating for movie 3 (M3) as:

Let's say we have the following parameters for user 1:

$$w^{(1)} = \begin{bmatrix} 5 \\ 0 \end{bmatrix}, b^{(1)} = 0$$

And the feature vector for movie 3 is given by:

$$x^{(3)} = \begin{bmatrix} 0.99 \\ 0 \end{bmatrix}$$

The predicted rating for movie 3 by user 1 is given by:

$$y^{(3,1)} = w^{(1)} x^{(3)} + b^{(1)} = \begin{bmatrix} 5 & 0 \end{bmatrix} \begin{bmatrix} 0.99 \\ 0 \end{bmatrix} + 0 = 4.95$$

## Cost Function
The notation used in this section is as follows:

- $n_u$ = number of users
- $n_m$ = number of movies
- $n$ = number of features
- $m^{(j)}$ = number of movies rated by user $j$
- $r(i,j)$ = 1 if user $j$ has rated movie $i$ and $r(i,j)$ = 0 otherwise
- $y^{(i,j)}$ = rating given by user $j$ to movie $i$ if $r(i,j)$ = 1
- $w^{(j)}, b^{(j)}$ = parameters for user $j$
- $x^{(i)}$ = feature vector for movie $i$

For user $j$, movie $i$, predict rating $y^{(i,j)} = w^{(j)}x^{(i)} + b^{(j)}$

The cost function (with regularization) for the recommender system is given by:

$$J(w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i:r(i,j)=1} ((w^{(j)}) x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{k=1}^n (w_k^{(j)})^2$$

Notice that the squared error is summed only over movies rated by user $j$ denoted by $r(i,j) = 1$. The regularization term is summed over all the features. We've also eliminated division by $m^{(j)}$ since it is a constant and does not affect the optimization.

To learn parameters $w^{(j)}, b^{(j)}$ for all the users, we need to minimize the cost function $J(w^{(j)}, b^{(j)})$ for all the users. The overall cost function is given by:

$$J(w^{(1)}, b^{(1)}, ..., w^{(n_u)}, b^{(n_u)}) = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} ((w^{(j)}) x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n (w_k^{(j)})^2$$

## Collaborative Filtering
In the table above, we have features for the movies that we can use to learn the parameters. What if we don't have the features? We can use collaborative filtering to learn the features. In collaborative filtering, we learn the features from the data. It is a technique used by recommender systems and is based on the assumption that people who agreed in the past will agree in the future, and that they will like similar kinds of items as they liked in the past.

Given the parameters $w^{(j)}, b^{(j)}$ for all the users, we can learn the features $x^{(i)}$ for all the movies by minimizing the cost function $J(x^{(1)}, ..., x^{(n_m)})$ given by:

$$J(x^{(1)}, ..., x^{(n_m)}) = \frac{1}{2} \sum_{i=1}^{n_m} \sum_{j:r(i,j)=1} ((w^{(j)}) x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n (x_k^{(i)})^2$$

Notice that in learning the features, we assume that the parameters $w^{(j)}, b^{(j)}$ are fixed. We can learn the parameters and the features simultaneously by minimizing the cost function $J(w^{(1)}, b^{(1)}, ..., w^{(n_u)}, b^{(n_u)}, x^{(1)}, ..., x^{(n_m)})$ given by:

$$J(w^{(1)}, b^{(1)}, ..., w^{(n_u)}, b^{(n_u)}, x^{(1)}, ..., x^{(n_m)}) = \frac{1}{2} \sum_{(i,j):r(i,j)=1} ((w^{(j)}) x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n (x_k^{(i)})^2$$

## Gradient Descent
In addition to the parameters $w^{(j)}, b^{(j)}$ for all the users, we also have the features $x^{(i)}$ for all the movies. We can learn the parameters and the features simultaneously by minimizing the cost function $J(w^{(1)}, b^{(1)}, ..., w^{(n_u)}, b^{(n_u)}, x^{(1)}, ..., x^{(n_m)})$ using gradient descent. The gradient descent update equations for the parameters and the features are given by:

$$w_i^{(j)} := w_i^{(j)} - \alpha \partial J(w, b, x) / \partial w_i^{(j)}$$

$$b^{(j)} := b^{(j)} - \alpha \partial J(w, b, x) / \partial b^{(j)}$$

$$x_k^{(i)} := x_k^{(i)} - \alpha \partial J(w, b, x) / \partial x_k^{(i)}$$

## Binary Labels
In the table above, we have ratings from 1 to 5. What if we have binary labels like "like" and "dislike"? We can use logistic regression to learn the parameters. The cost function (with regularization) for the ecommender system using logistic regression is given by:

$$J(w^{(j)}, b^{(j)}) = - \frac{1}{m^{(j)}} \sum_{i:r(i,j)=1} \left[ y^{(i,j)} \log (h_w(x^{(i)})) + (1 - y^{(i,j)}) \log (1 - h_w(x^{(i)})) \right] + \frac{\lambda}{2} \sum_{k=1}^n (w_k^{(j)})^2$$

where $h_w(x^{(i)}) = g(w^{(j)} x^{(i)} + b^{(j)})$ and $g(z) = 1 / (1 + e^{-z})$ is the sigmoid function. Notice that the regularization term is summed over all the features. We've also eliminated division by $m^{(j)}$ since it is a constant and does not affect the optimization.