import numpy as np

def estimate_gaussian(X):
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m, n = X.shape
    mu = np.zeros((n, 1))
    var = np.zeros((n, 1))
    mu = np.mean(X, axis=0).reshape(n, 1)
    var = np.var(X, axis=0).reshape(n, 1)
    return mu, var

def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    stepsize = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), stepsize):
        predictions = (p_val < epsilon).reshape(len(p_val), 1)
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = (2 * prec * rec) / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1

def detect_anomalies(X, X_val, y_val):
    """
    Finds the outliers in the dataset X using the validation 
    dataset X_val and y_val
    
    Args:
        X (ndarray):     (m, n) Data matrix
        X_val (ndarray): (m_val, n) Validation data matrix
        y_val (ndarray): (m_val, 1) Ground truth for validation set
        
    Returns:
        p (ndarray): (m, 1) Probability of each example being an outlier
    """
    mu, var = estimate_gaussian(X)
    p = np.zeros((X.shape[0], 1))
    p = np.prod(np.exp(-((X - mu.T) ** 2) / (2 * var.T)) / np.sqrt(2 * np.pi * var.T), axis=1).reshape(X.shape[0], 1)
    p_val = np.zeros((X_val.shape[0], 1))
    p_val = np.prod(np.exp(-((X_val - mu.T) ** 2) / (2 * var.T)) / np.sqrt(2 * np.pi * var.T), axis=1).reshape(X_val.shape[0], 1)
    epsilon, f1 = select_threshold(y_val, p_val)
    outliers = np.where(p < epsilon)
    return p, outliers