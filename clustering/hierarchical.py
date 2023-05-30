"""
HIERARCHICAL CLUSTERING

1. Start by assigning each item to a cluster, so that if you have N items, you now have N clusters, each containing just one item.
2. Find the closest (or most similar) pair of clusters and merge them into a single cluster, so that now you have one fewer cluster. 
   This is called agglomerative clustering.
3. Continue the process until all items are clustered into a single cluster of size N.
4. In practice, you stop when you have as many clusters as you wanted (say, 5).
5. At each step, you create a dendrogram, which is a tree that shows how each step led to the previous steps.
6. You have a stoping criteria, which is the number of clusters you want to have or the distance between clusters.

MEASURES OF DISTANCE
Linkage Metrics
* Single-linkage: consider the distance between the closest members of the two clusters.
* Complete-linkage: consider the distance between the farthest members of the two clusters.
* Average-linkage: consider the average distance between all possible pairs of members of the two clusters.

Hierachical clustering is deterministic, so it will always produce the same clusters given the same data and distance metric.
"""