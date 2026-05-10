"""Generate report figures from experiment result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qml_bc.plotting import plot_metric_summary, plot_paired_deltas
from qml_bc.results import compute_paired_deltas, summarize_by_model


DEFAULT_METRICS = (
    "balanced_accuracy",
    "roc_auc",
    "f1",
    "recall_positive",
)


def make_figures(
    results_path: str | Path,
    output_dir: str | Path,
    paired_model: str | None = None,
    baseline: str | None = None,
) -> list[Path]:
    """Create summary and optional paired-delta figures from a CSV."""

    results = pd.read_csv(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for metric in DEFAULT_METRICS:
        if metric not in results.columns:
            continue
        summary = summarize_by_model(results, metric)
        summary_path = output_dir / f"{metric}_summary.csv"
        summary.to_csv(summary_path, index=False)
        written.append(summary_path)
        written.append(
            plot_metric_summary(
                summary,
                metric,
                output_dir / f"{metric}_summary.png",
            )
        )

        if paired_model is not None and baseline is not None:
            delta = compute_paired_deltas(results, paired_model, baseline, metric)
            delta_path = output_dir / f"paired_delta_{metric}.csv"
            delta.to_csv(delta_path, index=False)
            written.append(delta_path)
            written.append(
                plot_paired_deltas(
                    delta,
                    metric,
                    output_dir / f"paired_delta_{metric}.png",
                )
            )

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Make report figures from results.")
    parser.add_argument("--results", required=True, help="Path to a results CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for figures.")
    parser.add_argument("--paired-model", help="Model name for paired deltas.")
    parser.add_argument("--baseline", help="Baseline model name for paired deltas.")
    args = parser.parse_args()

    written = make_figures(
        results_path=args.results,
        output_dir=args.output_dir,
        paired_model=args.paired_model,
        baseline=args.baseline,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

