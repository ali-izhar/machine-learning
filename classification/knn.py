"""
K Nearest Neighbors (KNN) classification algorithm

1. Choose a value for k
2. For each point in our data:
    2.1 Calculate the distance between the point and all other points
    2.2 Sort the distances and determine the k nearest neighbors
    2.3 Use the most popular response value from the k nearest neighbors as the predicted response value for the point
3. Return the predicted response values for each point

Choosing k
* Search for a good k
* Cross-validation
    # Split the data into training/test sets
    # Test the accuracy of the model using different values of k
    # Choose the k that corresponds to the lowest test error rate 
"""
import math

def KNearestClassify(training, testSet, label, k):
    """k-nearest neighbors classification"""
    # For each example in the test set, classify it using k-nearest neighbors
    results = []
    for e in testSet:
        # Get the k nearest neighbors
        neighbors = getNeighbors(training, e, k)
        # Get the majority vote
        majorityVote = getMajorityVote(neighbors, label)
        # Add predicted label to results
        results.append(majorityVote)
    return results

def getNeighbors(training, testInstance, k):
    """Get the k nearest neighbors"""
    distances = []
    for e in training:
        # Calculate the distance between the test instance and each training instance
        dist = euclideanDistance(testInstance, e)
        # Add the distance and the index of the training instance to an ordered collection
        distances.append((e, dist))
    # Sort the ordered collection of distances in ascending order
    distances.sort(key=lambda x: x[1])
    # Get the k nearest neighbors
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def getMajorityVote(neighbors, label):
    """Get the majority vote"""
    votes = {}
    for e in neighbors:
        # Get the label of the neighbor
        response = e[label]
        # Add the label to the votes dictionary
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    # Sort the votes dictionary in descending order
    sortedVotes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    # Return the label with the highest count
    return sortedVotes[0][0]

def euclideanDistance(instance1, instance2):
    """Calculate the Euclidean distance between two points"""
    distance = 0
    # Assume that the last element in each instance is the label
    for i in range(len(instance1)-1):
        distance += (instance1[i] - instance2[i])**2
    return math.sqrt(distance)