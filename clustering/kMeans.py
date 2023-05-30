"""
K-means clustering algorithm (k is the number of clusters)

1. Start by randomly choosing k centroids, one for each cluster you will create.
2. Create k clusters by assigning each example to closest centroid.
3. Compute k new centroids by averaging examples in each cluster.
4. Repeat steps 2 and 3 until centroids don't change.

Choosing k
* A priori knowledge about the application domain
    # there are 2 kinds of people in the world: k = 2
    # there are 5 types of backteria: k = 5

* Search for a good k
    # try different values of k and see which one is better
    # run hierarchical clustering on subset of the data to understand the cluster structure
      and then choose k

* Mitigating the effect of bad initial centroids
    # run k-means multiple times with different initial centroids
    # choose the best clustering (the one that minimizes the sum of squared errors)

K-means is non-deterministic, so it may produce different clusters depending on the initial choice of centroids.
"""
import random
from clustering import Cluster, Example


def kmeans(examples, k, verbose=False):
    # Get k randomly chosen initial centroids, create cluster for each
    initialCentroids = random.sample(examples, k)
    clusters = []
    for e in initialCentroids:
        clusters.append(Cluster([e]))
    
    # Iterate until centroids do not change
    converged = False
    numIterations = 0
    while not converged:
        numIterations += 1
        # Create a list containing k distinct empty lists
        newClusters = []
        for i in range(k):
            newClusters.append([])
        
        # Associate each example with closest centroid
        for e in examples:
            # Find the centroid closest to e
            smallestDistance = e.distance(clusters[0].getCentroid())
            index = 0
            for i in range(1, k):
                distance = e.distance(clusters[i].getCentroid())
                if distance < smallestDistance:
                    smallestDistance = distance
                    index = i
            # Add e to the list of examples for appropriate cluster
            newClusters[index].append(e)
        
        for c in newClusters: # Avoid having empty clusters
            if len(c) == 0:
                raise ValueError('Empty Cluster')
        
        # Update each cluster; check if a centroid has changed
        converged = True
        for i in range(k):
            if clusters[i].update(newClusters[i]) > 0.0:
                converged = False
        if verbose:
            print('Iteration #' + str(numIterations))
            for c in clusters:
                print(c)
            print('') # add blank line
    return clusters


def trykmeans(examples, numClusters, numTrials, verbose=False):
    """Calls kmeans numTrials times and returns the result with the lowest score."""
    best = kmeans(examples, numClusters, verbose)
    minScore = Cluster.agglomerativeScore(best)
    trial = 1
    while trial < numTrials:
        try:
            clusters = kmeans(examples, numClusters, verbose)
        except ValueError:
            continue # If failed, try again
        score = Cluster.agglomerativeScore(clusters)
        if score < minScore:
            best = clusters
            minScore = score
        trial += 1
    return best