"""Example of using KAN for time series anomaly detection."""

import torch
import numpy as np
from anomaly_detection.config import Args
from anomaly_detection.data import prepare_data, create_data_loaders
from anomaly_detection.models import KAN, FocalLoss
from anomaly_detection.train import Trainer
from anomaly_detection.visualization import plot_anomalies, plot_metrics


def main():
    # Set random seeds
    torch.manual_seed(Args.seed)
    np.random.seed(Args.seed)

    # Prepare data
    dataset, time_series, labels = prepare_data(Args)
    train_loader, val_loader, test_loader, test_indices = create_data_loaders(
        dataset, Args
    )

    # Initialize model and training components
    model = KAN(
        in_feat=1,
        hidden_feat=Args.hidden_size,
        out_feat=1,
        grid_feat=Args.grid_size,
        num_layers=Args.n_layers,
        dropout=Args.dropout,
    ).to(Args.device)

    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=1e-3,
        step_size_up=2000,
        mode="triangular",
        cycle_momentum=False,
    )

    # Train model
    trainer = Trainer(model, criterion, optimizer, scheduler, Args.device)
    optimal_threshold = trainer.train(
        train_loader, val_loader, Args.epochs, Args.early_stopping
    )

    # Evaluate and visualize results
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(Args.device)
            probs = torch.sigmoid(model(x_batch))
            preds = (probs > optimal_threshold).cpu().numpy()
            test_predictions.extend(preds)

    # Plot results
    plot_anomalies(time_series, labels, test_predictions)
    plot_metrics(labels, test_predictions)


if __name__ == "__main__":
    main()
