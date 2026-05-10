import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_run_experiment_cli_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "outputs" / "cli_smoke"
    config_path = tmp_path / "cli_smoke.yaml"
    config_path.write_text(
        f"""
experiment_name: cli_smoke
output_dir: {output_dir}

dataset:
  name: breast_cancer
  positive_label: malignant

splitting:
  outer_splits: 1
  test_size: 0.2
  inner_folds: 2
  seeds: [0]

metrics:
  - balanced_accuracy

models:
  - name: linear_svm_smoke
    type: linear_svm
    preprocessing:
      n_components: [4]
    params:
      C: [1.0]
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--config", str(config_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    results_path = output_dir / "results.csv"
    resolved_config_path = output_dir / "config_resolved.yaml"

    assert "Wrote results" in completed.stdout
    assert results_path.exists()
    assert resolved_config_path.exists()
    assert len(pd.read_csv(results_path)) == 1

