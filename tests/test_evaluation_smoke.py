from pathlib import Path

import numpy as np

from qml_bc.candidates import CandidateSpec
from qml_bc.config import ExperimentConfig, SplitConfig
from qml_bc.evaluation import build_model_for_candidate, run_experiment


def _classical_smoke_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="classical_smoke",
        dataset={"name": "breast_cancer", "positive_label": "malignant"},
        splitting=SplitConfig(
            outer_splits=2,
            test_size=0.2,
            inner_folds=2,
            seeds=(0, 1),
        ),
        models=[
            {
                "name": "linear_svm_smoke",
                "type": "linear_svm",
                "preprocessing": {"n_components": [4]},
                "params": {"C": [0.1, 1.0]},
            }
        ],
        metrics=("balanced_accuracy",),
        output_dir=tmp_path,
    )


def test_classical_evaluation_smoke_runs(tmp_path):
    results = run_experiment(_classical_smoke_config(tmp_path))

    assert len(results) == 2
    assert set(results["model_type"]) == {"linear_svm"}
    assert np.isfinite(results["balanced_accuracy"]).all()


def test_evaluation_returns_one_row_per_outer_split(tmp_path):
    results = run_experiment(_classical_smoke_config(tmp_path))

    assert list(results["split_id"]) == [0, 1]


def test_quantum_and_classical_share_model_builder_path():
    classical = CandidateSpec(
        candidate_id="classical",
        model_name="linear",
        model_type="linear_svm",
        model_params={"C": 1.0},
        preprocessing={"n_components": 2},
        quantum=None,
    )
    quantum = CandidateSpec(
        candidate_id="quantum",
        model_name="qk",
        model_type="quantum_kernel_svm",
        model_params={"C": 1.0},
        preprocessing={"n_components": 2},
        quantum={"encoding": "angle_y", "depth": 1, "alpha": 1.0},
    )

    assert build_model_for_candidate(classical).__class__.__name__ == "SVC"
    assert build_model_for_candidate(quantum).__class__.__name__ == "QuantumKernelSVC"

