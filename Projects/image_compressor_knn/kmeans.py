import numpy as np

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example in X.
    arguments:
        X (ndarray): (m, n) matrix of training examples
        centroids (ndarray): (K, n) matrix of centroids
    returns:
        idx (ndarray): (m, ) vector of centroid assignments (i.e. each entry in range [0, K-1])
    """
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros(m, dtype=int)
    # for each example in X, find the closest centroid
    for i in range(m):
        distance = []
        # for each centroid, compute the distance between the example and the centroid
        for j in range(K):
            # norm_ij = np.linalg.norm(X[i] - centroids[j])
            norm_ij = np.sum((X[i] - centroids[j])**2)
            distance.append(norm_ij)
        # find the index of the closest centroid
        idx[i] = np.argmin(distance)
    
    return idx


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the data points assigned to each centroid.
    arguments:
        X (ndarray): (m, n) matrix of training examples
        idx (ndarray): (m, ) vector of centroid assignments (i.e. each entry in range [0, K-1])
        K (int): number of clusters
    returns:
        centroids (ndarray): (K, n) new centroids computed
    """
    m, n = X.shape
    centroids = np.zeros((K, n))
    # for each centroid, compute the mean of the data points assigned to it
    for i in range(K):
        points = X[idx==i]
        centroids[i] = np.mean(points, axis=0)
    
    return centroids


def kmeans_random_init(X, K):
    """
    Randomly selects K data points from X to be the initial centroids.
    arguments:
        X (ndarray): (m, n) matrix of training examples
        K (int): number of clusters
    returns:
        centroids (ndarray): (K, n) matrix of initial centroids
    """

    # randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids


def run_kmeans(X, initial_centroids, max_iters):
    """
    Runs the K-means algorithm on data matrix X, where each row of X is a single example.
    """
    K = initial_centroids.shape[0]
    centroids = initial_centroids

    # run K-means
    for i in range(max_iters):
        print(f"Iteration {i+1}/{max_iters}...")
        # for each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        # given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)

    return centroids, idx