from pathlib import Path

import pytest

from qml_bc.config import load_config


def test_load_minimal_config():
    config = load_config(Path("configs/dev_smoke.yaml"))

    assert config.experiment_name == "dev_smoke"
    assert config.splitting.outer_splits == 2
    assert config.models[0]["type"] == "quantum_kernel_svm"


def test_missing_experiment_name_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
dataset: {}
splitting:
  outer_splits: 1
  test_size: 0.2
  inner_folds: 2
  seeds: [0]
models:
  - name: model
metrics: [accuracy]
output_dir: outputs/test
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="experiment_name"):
        load_config(path)


def test_invalid_split_config_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
experiment_name: bad
dataset: {}
splitting:
  outer_splits: 0
  test_size: 0.2
  inner_folds: 2
  seeds: [0]
models:
  - name: model
metrics: [accuracy]
output_dir: outputs/test
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outer_splits"):
        load_config(path)

