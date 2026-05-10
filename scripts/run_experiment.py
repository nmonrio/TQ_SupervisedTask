"""Command-line entry point for config-driven experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from qml_bc.config import load_config
from qml_bc.evaluation import run_experiment


def run_from_config(config_path: str | Path) -> Path:
    """Run an experiment config and write its reproducibility artifacts."""

    config_path = Path(config_path)
    config = load_config(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment(config)
    results_path = config.output_dir / "results.csv"
    results.to_csv(results_path, index=False)

    resolved_path = config.output_dir / "config_resolved.yaml"
    resolved_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a QML experiment from YAML.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML.")
    args = parser.parse_args()

    results_path = run_from_config(args.config)
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()

