import numpy as np
import pandas as pd
import pytest

from qml_bc.metrics import (
    centered_kernel_alignment,
    classification_metrics,
    paired_metric_delta,
)


def test_classification_metrics_known_values():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])

    metrics = classification_metrics(y_true, y_pred, y_score=None)

    assert metrics["recall_positive"] == 0.5
    assert metrics["balanced_accuracy"] == 0.75


def test_roc_auc_handles_scores():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_score = np.array([0.8, 0.7, -0.1, -0.2])

    metrics = classification_metrics(y_true, y_pred, y_score=y_score)

    assert metrics["roc_auc"] == 1.0


def test_kernel_alignment_is_scale_invariant():
    y = np.array([1, 1, 0, 0])
    K = np.array(
        [
            [1.0, 0.8, 0.1, 0.2],
            [0.8, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.7],
            [0.2, 0.1, 0.7, 1.0],
        ]
    )

    assert centered_kernel_alignment(K, y) == pytest.approx(
        centered_kernel_alignment(3.0 * K, y)
    )


def test_kernel_alignment_rejects_bad_shapes():
    with pytest.raises(ValueError, match="square"):
        centered_kernel_alignment(np.ones((2, 3)), np.array([0, 1]))


def test_paired_delta_matches_on_split_id():
    results = pd.DataFrame(
        {
            "split_id": [0, 1, 0, 1],
            "model_name": ["quantum", "quantum", "baseline", "baseline"],
            "balanced_accuracy": [0.8, 0.7, 0.75, 0.72],
        }
    )

    deltas = paired_metric_delta(results, "quantum", "baseline", "balanced_accuracy")

    assert list(deltas["delta"]) == pytest.approx([0.05, -0.02])

