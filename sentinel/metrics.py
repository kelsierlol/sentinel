"""Evaluation metrics for Sentinel."""

import numpy as np
from typing import Tuple


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Precision-Recall AUC.

    Args:
        scores: (N,) - Anomaly scores (higher = more anomalous)
        labels: (N,) - Binary labels (1 = anomaly, 0 = normal)

    Returns:
        PR-AUC score
    """
    scores = scores.flatten()
    labels = labels.flatten()

    # Filter out NaN scores
    valid_indices = ~np.isnan(scores)
    scores = scores[valid_indices]
    labels = labels[valid_indices]

    if len(np.unique(labels)) < 2:
        # PR-AUC is not well-defined if there's only one class
        return 0.0

    order = np.argsort(-scores)
    labels_sorted = labels[order]

    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)

    # Avoid division by zero
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(labels_sorted.sum(), 1e-12)

    # Add (0,0) and (1,0) points to PR curve
    precision = np.concatenate([[1.0], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [1.0]])

    trap = getattr(np, "trapezoid", np.trapz)  # Use np.trapz for older numpy
    return trap(precision, recall)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        scores: (N,) - Anomaly scores (higher = more anomalous)
        labels: (N,) - Binary labels (1 = anomaly, 0 = normal)

    Returns:
        AUROC score
    """
    scores = scores.flatten()
    labels = labels.flatten()

    # Filter out NaN scores
    valid_indices = ~np.isnan(scores)
    scores = scores[valid_indices]
    labels = labels[valid_indices]

    if len(np.unique(labels)) < 2:
        return 0.5  # Random classifier

    order = np.argsort(-scores)
    labels_sorted = labels[order]

    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)

    tpr = tp / np.maximum(labels_sorted.sum(), 1e-12)
    fpr = fp / np.maximum((1 - labels_sorted).sum(), 1e-12)

    # Add (0,0) and (1,1) points to ROC curve
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])

    trap = getattr(np, "trapezoid", np.trapz)
    return trap(tpr, fpr)


def f1_score(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """
    Compute F1 score at given threshold.

    Args:
        scores: (N,) - Anomaly scores
        labels: (N,) - Binary labels
        threshold: Score threshold for positive prediction

    Returns:
        F1 score
    """
    scores = scores.flatten()
    labels = labels.flatten()

    preds = (scores >= threshold).astype(int)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)

    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return float(f1)


def iou_at_k(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute IoU at top-k coverage (k = fraction of positive labels).

    Args:
        scores: (N,) - Anomaly scores
        labels: (N,) - Binary labels

    Returns:
        IoU score
    """
    scores = scores.flatten()
    labels = labels.flatten()

    coverage = labels.mean()
    if coverage <= 0:
        return 0.0

    thresh = np.quantile(scores, 1.0 - coverage)
    preds = (scores >= thresh).astype(int)

    intersection = np.sum((preds == 1) & (labels == 1))
    union = np.sum((preds == 1) | (labels == 1))

    return float(intersection / max(union, 1))


def fpr_at_recall(scores: np.ndarray, labels: np.ndarray, recall_target: float = 0.9) -> float:
    """
    Compute False Positive Rate at target recall.

    Args:
        scores: (N,) - Anomaly scores
        labels: (N,) - Binary labels
        recall_target: Target recall (default 0.9)

    Returns:
        FPR at target recall
    """
    scores = scores.flatten()
    labels = labels.flatten()

    order = np.argsort(-scores)
    labels_sorted = labels[order]

    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)

    recall = tp / max(labels_sorted.sum(), 1e-12)

    idx = np.searchsorted(recall, recall_target, side="left")
    if idx >= len(fp):
        idx = len(fp) - 1

    return float(fp[idx] / max((1 - labels_sorted).sum(), 1e-12))


def optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.05
) -> Tuple[float, float]:
    """
    Find threshold that achieves target FPR on normal (label=0) samples.

    Args:
        scores: (N,) - Anomaly scores
        labels: (N,) - Binary labels
        target_fpr: Target false positive rate (default 0.05)

    Returns:
        (threshold, actual_fpr)
    """
    scores = scores.flatten()
    labels = labels.flatten()

    # Get scores for normal samples only
    normal_scores = scores[labels == 0]

    if len(normal_scores) == 0:
        return 0.5, 0.0

    # Threshold at (1 - target_fpr) quantile
    threshold = np.quantile(normal_scores, 1.0 - target_fpr)

    # Compute actual FPR
    actual_fpr = (normal_scores >= threshold).mean()

    return float(threshold), float(actual_fpr)
