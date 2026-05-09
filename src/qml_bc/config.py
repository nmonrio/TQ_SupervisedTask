"""Configuration loading for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SplitConfig:
    outer_splits: int
    test_size: float
    inner_folds: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    dataset: dict[str, Any]
    splitting: SplitConfig
    models: list[dict[str, Any]]
    metrics: tuple[str, ...]
    output_dir: Path


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("config file must contain a YAML mapping")

    try:
        experiment_name = raw["experiment_name"]
        dataset = raw["dataset"]
        splitting = _parse_split_config(raw["splitting"])
        models = raw["models"]
        metrics = tuple(raw["metrics"])
        output_dir = Path(raw["output_dir"])
    except KeyError as exc:
        raise ValueError(f"missing required config key: {exc.args[0]}") from exc

    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError("experiment_name must be a non-empty string")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a mapping")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    if not metrics:
        raise ValueError("metrics must be non-empty")

    return ExperimentConfig(
        experiment_name=experiment_name,
        dataset=dataset,
        splitting=splitting,
        models=models,
        metrics=metrics,
        output_dir=output_dir,
    )


def _parse_split_config(raw: Any) -> SplitConfig:
    if not isinstance(raw, dict):
        raise ValueError("splitting must be a mapping")

    try:
        outer_splits = int(raw["outer_splits"])
        test_size = float(raw["test_size"])
        inner_folds = int(raw["inner_folds"])
        seeds = tuple(int(seed) for seed in raw["seeds"])
    except KeyError as exc:
        raise ValueError(f"missing required splitting key: {exc.args[0]}") from exc
    except TypeError as exc:
        raise ValueError("splitting.seeds must be an iterable of integers") from exc

    if outer_splits < 1:
        raise ValueError("outer_splits must be >= 1")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if inner_folds < 2:
        raise ValueError("inner_folds must be >= 2")
    if len(seeds) < outer_splits:
        raise ValueError("splitting.seeds must contain at least outer_splits values")

    return SplitConfig(
        outer_splits=outer_splits,
        test_size=test_size,
        inner_folds=inner_folds,
        seeds=seeds,
    )

