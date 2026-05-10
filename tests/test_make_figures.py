import pandas as pd

from scripts.make_figures import make_figures


def test_make_figures_writes_summary_outputs(tmp_path):
    results_path = tmp_path / "results.csv"
    results = pd.DataFrame(
        {
            "split_id": [0, 1, 0, 1],
            "model_name": ["quantum", "quantum", "baseline", "baseline"],
            "balanced_accuracy": [0.8, 0.7, 0.75, 0.72],
            "roc_auc": [0.9, 0.85, 0.88, 0.84],
        }
    )
    results.to_csv(results_path, index=False)

    written = make_figures(results_path, tmp_path / "figures")

    names = {path.name for path in written}
    assert "balanced_accuracy_summary.csv" in names
    assert "balanced_accuracy_summary.png" in names
    assert "roc_auc_summary.csv" in names
    assert "roc_auc_summary.png" in names


def test_make_figures_writes_paired_delta_outputs(tmp_path):
    results_path = tmp_path / "results.csv"
    results = pd.DataFrame(
        {
            "split_id": [0, 1, 0, 1],
            "model_name": ["quantum", "quantum", "baseline", "baseline"],
            "balanced_accuracy": [0.8, 0.7, 0.75, 0.72],
        }
    )
    results.to_csv(results_path, index=False)

    written = make_figures(
        results_path,
        tmp_path / "figures",
        paired_model="quantum",
        baseline="baseline",
    )

    names = {path.name for path in written}
    assert "paired_delta_balanced_accuracy.csv" in names
    assert "paired_delta_balanced_accuracy.png" in names

