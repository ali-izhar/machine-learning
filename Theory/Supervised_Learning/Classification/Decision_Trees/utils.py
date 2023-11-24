import numpy as np
import matplotlib.pyplot as plt


def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def split_indices(X, index_feature):
    """
    Given a dataset and a index feature, return two lists for the two split nodes, 
    the left node has the animals that have that feature = 1 and the right node 
    those that have the feature = 0.
    """
    left_indices = []
    right_indices = []
    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)


def weighted_entropy(X, y, left_indices, right_indices):
    """
    Take the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    n = X.shape[0]
    w_left = len(left_indices) / n
    w_right = len(right_indices) / n
    p_left = sum(y[left_indices]) / len(left_indices)
    p_right = sum(y[right_indices]) / len(right_indices)

    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy


def information_gain(X, y, left_indices, right_indices):
    """
    Take the splitted dataset, the indices we chose to split and returns the information gain.
    """
    root_entropy = entropy(sum(y) / len(y))
    weighted_entropy = weighted_entropy(X, y, left_indices, right_indices)
    return root_entropy - weighted_entropy


def best_split(X, y):
    """
    Given a dataset, returns the best feature to split the dataset and the indices of the splitted dataset.
    """
    best_feature = None
    best_gain = 0
    best_left_indices = []
    best_right_indices = []

    for i in range(X.shape[1]):
        left_indices, right_indices = split_indices(X, i)
        gain = information_gain(X, y, left_indices, right_indices)
        if gain > best_gain:
            best_gain = gain
            best_feature = i
            best_left_indices = left_indices
            best_right_indices = right_indices

    return best_feature, best_left_indices, best_right_indices


def plot_entropy():
    x = np.linspace(0, 1, 100)
    y = [entropy(p) for p in x]
    plt.plot(x, y)
    plt.xlabel('p')
    plt.ylabel('H(p)')
    plt.show()


def plot_weighted_entropy():
    x = np.linspace(0, 1, 100)
    y = [weighted_entropy(x, [1] * 100, [1] * 50, [1] * 50) for x in x]
    plt.plot(x, y)
    plt.xlabel('p')
    plt.ylabel('H(p)')
    plt.show()


def plot_information_gain():
    x = np.linspace(0, 1, 100)
    y = [information_gain(x, [1] * 100, [1] * 50, [1] * 50) for x in x]
    plt.plot(x, y)
    plt.xlabel('p')
    plt.ylabel('IG(p)')
    plt.show()