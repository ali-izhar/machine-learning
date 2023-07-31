# K-means Clustering: A Quick Walkthrough

## Introduction
K-means clustering is an iterative, unsupervised machine learning algorithm that separates a given dataset into K clusters. It does this by minimizing the **in-cluster variance**, which is computed as the sum of the Euclidean distances between each data point and the centroid of its assigned cluster.

## The Process
Assume we have a dataset of 30 unlabeled training examples. These examples are plotted as points on a 2D grid. We want to apply K-means clustering to this dataset.

Here's how it works:

### Step 1: Initialize Cluster Centroids
To begin, K-means randomly selects $K$ data points as the initial centroids. In this example, we want to find two clusters, so $K=2$.

### Step 2: Assign Points to Cluster Centroids
Now, for each data point in the dataset, the algorithm assigns it to the nearest centroid. This is done by calculating the Euclidean distance between the data point and each centroid. The data point is assigned to the centroid to which it has the shortest distance.

### Step 3: Recalculate Cluster Centroids
Next, K-means recalculates the centroids by taking the mean of all data points assigned to each centroid's cluster. This essentially shifts the centroids to the average location of their respective clusters.

### Step 4: Repeat Steps 2 and 3
The algorithm repeats the assignment and update steps until the positions of the centroids stabilize, and the clusters become consistent. This means that the assignment of data points to clusters no longer changes.

Throughout this iterative process, some data points may change their cluster membership as the centroids shift. The algorithm continues to run until it reaches a point where no changes are observed in the assignment of points to clusters or the positions of centroids.

## K-means Algorithm
Here's a pseudocode implementation of the K-means algorithm:

```
Randomly initialize K cluster centroids (μ1, μ2, ..., μK) ∈ ℝn

Repeat {
    # assign points to clusters
    for i = 1 to m {
        c(i) := index (from 1 to K) of cluster centroid closest to x(i)
        # min ||x(i) - μk||^2
    }

    # move centroids to average of assigned points
    for k = 1 to K {
        μk := average (mean) of points assigned to cluster k
    }
}
```

Corner cases:
- If a cluster ends up with no points assigned to it, then the centroid is not updated. This is because the average of zero points is undefined. In this case, it's best to randomly reinitialize the centroid or remove the cluster altogether and run the algorithm again with $K-1$ clusters.