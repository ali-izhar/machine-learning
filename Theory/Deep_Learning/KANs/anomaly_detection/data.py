"""Data handling utilities for KAN Anomaly Detection"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # type: ignore
from sklearn.model_selection import train_test_split


class TimeSeriesAnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, time_series, labels, window_size=20, step_size=10):
        self.time_series = time_series
        self.labels = labels
        self.window_size = window_size
        self.step_size = step_size
        self.sample_indices = list(
            range(0, len(time_series) - window_size + 1, step_size)
        )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        i = self.sample_indices[idx]
        window = self.time_series[i : i + self.window_size]
        window_labels = self.labels[i : i + self.window_size]
        x = torch.tensor(window, dtype=torch.float).unsqueeze(-1)
        y = torch.tensor(1.0 if window_labels.any() else 0.0, dtype=torch.float)
        return x, y


def generate_sine_wave_with_anomalies(
    length=5000, anomaly_fraction=0.1, window_size=20, step_size=10
):
    x = np.linspace(0, 100 * np.pi, length)
    y = np.sin(x)
    labels = np.zeros(length)
    window_centers = list(range(window_size // 2, length - window_size // 2, step_size))
    num_anomalies = int(len(window_centers) * anomaly_fraction)
    anomaly_centers = np.random.choice(window_centers, num_anomalies, replace=False)

    for center in anomaly_centers:
        anomaly_type = np.random.choice(["point", "contextual", "collective"])
        if anomaly_type == "point":
            y[center] += np.random.normal(0, 10)
            labels[center] = 1
        elif anomaly_type == "contextual":
            y[center] = y[center] * np.random.uniform(1.5, 2.0)
            labels[center] = 1
        elif anomaly_type == "collective":
            start = max(0, center - window_size // 2)
            end = min(length, center + window_size // 2)
            y[start:end] += np.random.normal(0, 5, size=end - start)
            labels[start:end] = 1

    return y, labels


def prepare_data(args):
    # Generate data
    time_series, labels = generate_sine_wave_with_anomalies(
        length=5000,
        anomaly_fraction=args.anomaly_fraction,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    # Scale data
    scaler = StandardScaler()
    time_series = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()

    # Create dataset
    dataset = TimeSeriesAnomalyDataset(
        time_series, labels, window_size=args.window_size, step_size=args.step_size
    )

    return dataset, time_series, labels


def create_data_loaders(dataset, args):
    # Split indices
    train_indices, val_indices, test_indices = stratified_split(dataset, seed=args.seed)

    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Apply SMOTE to training data
    X_train = [x.numpy().flatten() for x, _ in train_dataset]
    y_train = [int(y.item()) for _, y in train_dataset]
    smote = SMOTE(random_state=args.seed)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Create balanced dataset
    balanced_train_dataset = ResampledDataset(X_resampled, y_resampled)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        balanced_train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, test_indices


class ResampledDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = [torch.tensor(x, dtype=torch.float).view(-1, 1) for x in X]
        self.y = [torch.tensor(label, dtype=torch.float) for label in y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def stratified_split(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    labels = [y.item() for _, y in dataset]
    train_val_indices, test_indices = train_test_split(
        np.arange(len(labels)), test_size=test_ratio, stratify=labels, random_state=seed
    )
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_relative_ratio,
        stratify=[labels[i] for i in train_val_indices],
        random_state=seed,
    )
    return train_indices, val_indices, test_indices
