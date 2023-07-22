import numpy as np

class SoftmaxRegressor:

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    def cross_entropy(self, y, y_hat):
        return - np.sum(y * np.log(y_hat)) / y.shape[0]