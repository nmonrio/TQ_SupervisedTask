"""Metrics used for model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    positive_label: int = 1,
) -> dict[str, float]:
    """Compute the core classification metrics for this task."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label)),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=positive_label)),
        "roc_auc": float("nan"),
    }

    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_score)))

    return metrics


def centered_kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Compute centered kernel-target alignment for binary labels."""

    K = np.asarray(K, dtype=float)
    y = np.asarray(y)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")
    if K.shape[0] != len(y):
        raise ValueError("K and y must contain the same number of samples")

    labels = np.where(y == 1, 1.0, -1.0)
    yy = np.outer(labels, labels)
    Kc = _center_matrix(K)
    Yc = _center_matrix(yy)
    denominator = np.linalg.norm(Kc, ord="fro") * np.linalg.norm(Yc, ord="fro")
    if denominator == 0.0:
        return 0.0
    return float(np.sum(Kc * Yc) / denominator)


def paired_metric_delta(
    results: pd.DataFrame,
    model_name: str,
    baseline_name: str,
    metric: str,
) -> pd.DataFrame:
    """Return split-wise metric deltas between a model and a baseline."""

    required = {"split_id", "model_name", metric}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results missing required columns: {sorted(missing)}")

    model = results.loc[results["model_name"] == model_name, ["split_id", metric]]
    base = results.loc[results["model_name"] == baseline_name, ["split_id", metric]]
    paired = model.merge(base, on="split_id", suffixes=("_model", "_baseline"))
    if len(paired) != len(model) or len(paired) != len(base):
        raise ValueError("model and baseline results must have matching split_id values")

    paired["delta"] = paired[f"{metric}_model"] - paired[f"{metric}_baseline"]
    return paired[["split_id", f"{metric}_model", f"{metric}_baseline", "delta"]]


def _center_matrix(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ M @ H

