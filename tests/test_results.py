import pandas as pd
import pytest

from qml_bc.results import compute_paired_deltas, summarize_by_model


def test_summarize_by_model_known_values():
    results = pd.DataFrame(
        {
            "model_name": ["a", "a", "b", "b"],
            "balanced_accuracy": [0.8, 1.0, 0.5, 0.7],
        }
    )

    summary = summarize_by_model(results, "balanced_accuracy")
    row_a = summary.loc[summary["model_name"] == "a"].iloc[0]

    assert row_a["mean"] == pytest.approx(0.9)
    assert row_a["n_splits"] == 2
    assert row_a["sem"] == pytest.approx(0.1)


def test_compute_paired_deltas_matches_split_ids():
    results = pd.DataFrame(
        {
            "split_id": [0, 1, 0, 1],
            "model_name": ["quantum", "quantum", "baseline", "baseline"],
            "balanced_accuracy": [0.8, 0.7, 0.75, 0.72],
        }
    )

    deltas = compute_paired_deltas(
        results,
        model_name="quantum",
        baseline_name="baseline",
        metric="balanced_accuracy",
    )

    assert list(deltas["delta"]) == pytest.approx([0.05, -0.02])


def test_compute_paired_deltas_raises_if_split_missing():
    results = pd.DataFrame(
        {
            "split_id": [0, 1, 0],
            "model_name": ["quantum", "quantum", "baseline"],
            "balanced_accuracy": [0.8, 0.7, 0.75],
        }
    )

    with pytest.raises(ValueError, match="matching split_id"):
        compute_paired_deltas(
            results,
            model_name="quantum",
            baseline_name="baseline",
            metric="balanced_accuracy",
        )

