from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _maybe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _maybe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float | None]:
    y_true = np.asarray(y_true).astype(np.uint8).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_pred = (y_score >= threshold).astype(np.uint8)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": _maybe_auc(y_true, y_score),
        "average_precision": _maybe_average_precision(y_true, y_score),
    }


def segmentation_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true).astype(np.uint8).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_pred = (y_score >= threshold).astype(np.uint8)
    intersection = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    union = int(np.logical_or(y_true == 1, y_pred == 1).sum())
    return {
        "base_accuracy": float(accuracy_score(y_true, y_pred)),
        "base_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "base_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "base_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "base_iou": float(intersection / union) if union else 0.0,
        "base_auroc": _maybe_auc(y_true, y_score),
        "base_average_precision": _maybe_average_precision(y_true, y_score),
    }
