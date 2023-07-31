def k_means(data, k):
    # Initialize centroids
    centroids = initialize_centroids(data, k)
    
    # Initialize clusters
    clusters = [[] for _ in range(k)]
    
    # Initialize old_centroids
    old_centroids = None
    
    # Iterate until centroids converge
    while not has_converged(centroids, old_centroids):
        # Assign data points to clusters
        clusters = assign_points(data, centroids)
        
        # Save old centroids
        old_centroids = centroids
        
        # Recalculate centroids
        centroids = recalculate_centroids(clusters)
        
    return clusters