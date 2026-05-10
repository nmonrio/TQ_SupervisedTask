"""Utilities for turning split-level results into report tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_by_model(
    results: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Summarize a metric by model with standard errors and normal CIs."""

    required = {"model_name", metric}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results missing required columns: {sorted(missing)}")

    rows = []
    for model_name, group in results.groupby("model_name", sort=True):
        values = group[metric].astype(float).to_numpy()
        n_splits = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n_splits > 1 else 0.0
        sem = float(std / np.sqrt(n_splits)) if n_splits > 0 else float("nan")
        ci_width = 1.96 * sem
        rows.append(
            {
                "model_name": model_name,
                "metric": metric,
                "mean": mean,
                "std": std,
                "sem": sem,
                "n_splits": n_splits,
                "ci95_low": mean - ci_width,
                "ci95_high": mean + ci_width,
            }
        )

    return pd.DataFrame(rows)


def compute_paired_deltas(
    results: pd.DataFrame,
    model_name: str,
    baseline_name: str,
    metric: str,
) -> pd.DataFrame:
    """Compute split-wise metric deltas for matched model/baseline rows."""

    required = {"split_id", "model_name", metric}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results missing required columns: {sorted(missing)}")

    model = results.loc[results["model_name"] == model_name, ["split_id", metric]]
    baseline = results.loc[
        results["model_name"] == baseline_name,
        ["split_id", metric],
    ]

    if model.empty:
        raise ValueError(f"no rows found for model_name={model_name!r}")
    if baseline.empty:
        raise ValueError(f"no rows found for baseline_name={baseline_name!r}")

    paired = model.merge(baseline, on="split_id", suffixes=("_model", "_baseline"))
    if len(paired) != len(model) or len(paired) != len(baseline):
        raise ValueError("model and baseline results must have matching split_id values")

    paired["delta"] = paired[f"{metric}_model"] - paired[f"{metric}_baseline"]
    return paired.sort_values("split_id").reset_index(drop=True)

