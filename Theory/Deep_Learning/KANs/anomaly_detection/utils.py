"""Utility functions for KAN Anomaly Detection"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def evaluate_metrics(true_labels, pred_labels, pred_probs):
    """Calculate various evaluation metrics."""
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    roc_auc_val = roc_auc_score(true_labels, pred_probs)
    return precision, recall, f1, roc_auc_val


def find_optimal_threshold(probs, labels):
    """Find the optimal threshold based on F1 score."""
    precision_vals, recall_vals, thresholds = precision_recall_curve(labels, probs)
    f1_scores = (
        2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
    )
    optimal_idx = np.argmax(f1_scores)
    if optimal_idx < len(thresholds):
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5
    optimal_f1 = f1_scores[optimal_idx]
    return optimal_threshold, optimal_f1


def aggregate_predictions(indices, preds, window_size, total_length):
    """Aggregate window-based predictions into a single time series."""
    aggregated = np.zeros(total_length, dtype=float)
    counts = np.zeros(total_length, dtype=float)

    for idx, pred in zip(indices, preds):
        start = idx
        end = idx + window_size
        if end > total_length:
            end = total_length
        aggregated[start:end] += pred
        counts[start:end] += 1

    counts[counts == 0] = 1
    averaged = aggregated / counts
    return (averaged > 0.5).astype(int)
