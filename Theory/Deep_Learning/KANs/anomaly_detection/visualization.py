"""Visualization utilities for KAN Anomaly Detection"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_anomalies(time_series, labels, preds, start=0, end=1000, save_path=None):
    """Plot time series with true and predicted anomalies."""
    plt.figure(figsize=(15, 5))

    # Plot time series
    plt.plot(time_series[start:end], label="Time Series", color="blue", alpha=0.7)

    # Plot true anomalies
    true_anomalies = np.arange(start, end)[labels[start:end] == 1]
    plt.scatter(
        true_anomalies,
        time_series[start:end][labels[start:end] == 1],
        color="red",
        label="True Anomalies",
        alpha=0.6,
    )

    # Plot predicted anomalies
    pred_anomalies = np.arange(start, end)[preds[start:end] == 1]
    plt.scatter(
        pred_anomalies,
        time_series[start:end][preds[start:end] == 1],
        color="orange",
        marker="x",
        label="Predicted Anomalies",
        alpha=0.8,
    )

    plt.legend()
    plt.title("Anomaly Detection Results")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Value")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_metrics(true_labels, pred_probs, save_path=None):
    """Plot ROC and Precision-Recall curves."""
    # Calculate curves
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    precision_vals, recall_vals, _ = precision_recall_curve(true_labels, pred_probs)

    # Calculate AUC scores
    roc_auc_val = auc(fpr, tpr)
    pr_auc_val = auc(recall_vals, precision_vals)

    # Create plot
    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall_vals, precision_vals, label=f"PR Curve (AUC = {pr_auc_val:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
