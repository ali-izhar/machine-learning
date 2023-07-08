import numpy as np

def scale(data: np.ndarray, method: str) -> np.ndarray:
    """
    Scales the data using the specified method. 
    Supported methods: min-max, mean-normalization, z-score
    :param data: data to be scaled
    :param method: method to use for scaling
    :return: scaled data
    """
    match method:
        case "min-max":
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        case "mean-normalization":
            return (data - np.mean(data)) / (np.max(data) - np.min(data))
        case "z-score":
            return (data - np.mean(data)) / np.std(data)
        case _:
            raise ValueError(f"Invalid method: {method}\nSupported methods: min-max, mean-normalization, z-score")
        
