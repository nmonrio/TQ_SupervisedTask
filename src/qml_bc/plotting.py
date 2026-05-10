"""Plotting helpers for report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def configure_plot_style() -> None:
    """Use a LaTeX-like serif style without requiring a TeX installation."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


configure_plot_style()


def plot_metric_summary(
    summary_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
) -> Path:
    """Plot model means with 95% confidence intervals."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = summary_df.loc[summary_df["metric"] == metric].sort_values("mean")

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    yerr = [
        data["mean"] - data["ci95_low"],
        data["ci95_high"] - data["mean"],
    ]
    ax.barh(data["model_name"], data["mean"], xerr=yerr, color="#4C78A8")
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_ylabel("")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_paired_deltas(
    delta_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
) -> Path:
    """Plot split-wise paired deltas against a zero reference line."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = delta_df.sort_values("split_id")

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.plot(data["split_id"], data["delta"], marker="o", color="#F58518")
    ax.set_xlabel("split id")
    ax.set_ylabel(f"delta {metric.replace('_', ' ')}")
    ax.grid(alpha=0.25)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_kernel_alignment(
    alignment_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot kernel-target alignment values by model or encoding."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_col = "model_name" if "model_name" in alignment_df.columns else "encoding_name"
    data = alignment_df.sort_values("alignment")

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.barh(data[label_col], data["alignment"], color="#54A24B")
    ax.set_xlabel("centered kernel alignment")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
