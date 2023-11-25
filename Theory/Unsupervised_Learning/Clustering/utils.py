import pandas as pd
import matplotlib.pyplot as plt

from kmeans import kmeans_random_init, run_kmeans

def plot_clusters(X, idx, centroids, old_centroids, K):
    """
    Plots the data points with colors assigned to each centroid.
    arguments:
        X (ndarray): (m, n) matrix of training examples
        idx (ndarray): (m, ) vector of centroid assignments (i.e. each entry in range [0, K-1])
        centroids (ndarray): (K, n) matrix of centroids
        old_centroids (ndarray): (K, n) matrix of old centroids
        K (int): number of clusters
        num_iters (int): number of iterations
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='viridis', alpha=0.8)
    
    # Draw lines between old and new centroids
    for i in range(K):
        plt.plot([old_centroids[i, 0], centroids[i, 0]], [old_centroids[i, 1], centroids[i, 1]], 'k-')
    
    # Use different marker for centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, alpha=1, marker='*', label='Centroids')
    
    # Annotate the centroids
    for i, centroid in enumerate(centroids):
        plt.annotate(i, (centroid[0], centroid[1]), color='red')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(False)
    plt.show()


if __name__ == '__main__':
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)

    # the dataset has 150 rows and 5 columns
    # the first 4 columns are the features and the last column is the label

    # features (sepal-length, sepal-width, petal-length, petal-width)
    # select all rows and all columns except the last column, convert to numpy array (using .values)
    X = dataset.iloc[:, :-1].values
    
    # label (class)
    # select all rows and the last column, convert to numpy array (using .values)
    y = dataset.iloc[:, 4].values
    
    K = 4
    m = X.shape[0]
    n = X.shape[1]
    max_iters = 10

    initial_centroids = kmeans_random_init(X, K)
    centroids, idx = run_kmeans(X, initial_centroids, max_iters)

    # plot the final clusters
    plot_clusters(X, idx, centroids, initial_centroids, K)