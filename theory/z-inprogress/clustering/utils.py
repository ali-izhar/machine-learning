def dissimilarity(clusters):
    """Assumes clusters a list of clusters
    Returns a measure of the total dissimilarity of the
    clusters in the list"""
    totDist = 0
    for c in clusters:
        totDist += c.variability()
    return totDist


def printClustering(clustering):
    """Assumes: clustering is a sequence of clusters
       Prints information about each cluster
       Returns list of fraction of variance explained by each cluster"""
    numClusters = len(clustering)
    print('Number of clusters =', numClusters)
    for c in clustering:
        print(c)
    print('Total dissimilarity =', dissimilarity(clustering))


def testClustering(clustering, numClusters, numTrials):
    """Assumes: clustering a function that partitions a list of
         data into clusters using some algorithm
       numClusters an int >= 1
       numTrials an int > 0
     Prints, for each trial, the dissimilarity of the
         resulting clustering and the fraction of data points
         in each cluster"""
    totDist = 0.0
    for _ in range(numTrials):
        clusters = clustering(numClusters)
        totDist += dissimilarity(clusters)
    print('Mean dissimilarity of', numClusters,
          'clusters =', totDist/numTrials)