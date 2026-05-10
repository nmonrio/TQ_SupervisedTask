import pandas as pd
from matplotlib import pyplot as plt

from qml_bc.plotting import (
    plot_kernel_alignment,
    plot_metric_summary,
    plot_paired_deltas,
)


def test_plot_style_uses_latex_like_serif_font():
    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["mathtext.fontset"] == "cm"


def test_plot_metric_summary_creates_file(tmp_path):
    summary = pd.DataFrame(
        {
            "model_name": ["a", "b"],
            "metric": ["balanced_accuracy", "balanced_accuracy"],
            "mean": [0.8, 0.9],
            "ci95_low": [0.7, 0.85],
            "ci95_high": [0.9, 0.95],
        }
    )

    output_path = plot_metric_summary(
        summary,
        "balanced_accuracy",
        tmp_path / "summary.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_paired_deltas_creates_file(tmp_path):
    deltas = pd.DataFrame(
        {
            "split_id": [0, 1, 2],
            "delta": [0.01, -0.02, 0.03],
        }
    )

    output_path = plot_paired_deltas(
        deltas,
        "balanced_accuracy",
        tmp_path / "deltas.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_kernel_alignment_creates_file(tmp_path):
    alignment = pd.DataFrame(
        {
            "model_name": ["qk_a", "qk_b"],
            "alignment": [0.1, 0.2],
        }
    )

    output_path = plot_kernel_alignment(alignment, tmp_path / "alignment.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
