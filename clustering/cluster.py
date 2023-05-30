import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

__all__ = ['Cluster', 'Example']

class Cluster(object):
    def __init__(self, examples):
        """Assumes examples a non-empty list of Examples"""
        self.examples = examples
        self.centroid = self.computeCentroid()

    def update(self, examples):
        """Assume examples is a non-empty list of Examples
        Replace examples; return amount centroid has changed"""
        oldCentroid = self.centroid
        self.examples = examples
        self.centroid = self.computeCentroid()
        return oldCentroid.distance(self.centroid)
    
    def computeCentroid(self):
        vals = np.array([0]*self.examples[0].dimensionality())
        for e in self.examples:
            vals += e.getFeatures()
        centroid = Example('centroid', vals/len(self.examples))
        return centroid
    
    def getCentroid(self):
        return self.centroid
    
    def variability(self):
        totDist = 0.0
        for e in self.examples:
            totDist += (e.distance(self.centroid))**2
        return totDist
    
    def members(self):
        for e in self.examples:
            yield e

    def __str__(self):
        names = []
        for e in self.examples:
            names.append(e.getName())
        names.sort()
        result = 'Cluster with centroid '\
               + str(self.centroid.getFeatures()) + ' contains:\n  '
        for e in names:
            result = result + e + ', '
        return result[:-2]


class Example(object):
    def __init__(self, name, features, label = None):
        """Assumes features is an array of floats
        Construct a labeled example"""
        self.name = name
        self.features = features
        self.label = label

    def dimensionality(self):
        """Returns the dimensionality of the example"""
        return len(self.features)

    def getFeatures(self):
        """Returns a copy of the features of the example"""
        return self.features[:]

    def getLabel(self):
        """Returns the label of the example"""
        return self.label

    def getName(self):
        """Returns the name of the example"""
        return self.name

    def distance(self, other):
        """Assumes other an example
        Returns the Euclidean distance between feature vectors
        of self and other"""
        return euclidean_distances(self.features, other.getFeatures())

    def __str__(self):
        return self.name + ':' + str(self.features) + ':'\
               + str(self.label)