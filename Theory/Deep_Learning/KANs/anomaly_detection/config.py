"""Configuration for KAN Anomaly Detection"""

import torch


class Args:
    path = "./data/"
    dropout = 0.3
    hidden_size = 128
    grid_size = 50  # The number of control points in each dimension
    n_layers = 2  # The number of layers
    epochs = 200
    early_stopping = 30
    seed = 42
    lr = 1e-3
    window_size = 20  # Window size is the number of time steps in the input sequence
    step_size = 10  # Step size is the number of time steps between consecutive inputs
    batch_size = 32  # Batch size is the number of input sequences in a batch
    anomaly_fraction = 0.1  # Fraction of anomalies in the dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
