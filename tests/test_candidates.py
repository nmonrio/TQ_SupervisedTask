from qml_bc.candidates import expand_candidates
from qml_bc.config import ExperimentConfig, SplitConfig, load_config


def _config(models):
    return ExperimentConfig(
        experiment_name="test",
        dataset={"name": "breast_cancer"},
        splitting=SplitConfig(
            outer_splits=1,
            test_size=0.2,
            inner_folds=2,
            seeds=(0,),
        ),
        models=models,
        metrics=("balanced_accuracy",),
        output_dir="outputs/test",
    )


def test_expand_candidates_cartesian_product():
    config = _config(
        [
            {
                "name": "rbf_svm_pca",
                "type": "rbf_svm",
                "preprocessing": {"n_components": [4, 6, 8]},
                "params": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
            }
        ]
    )

    candidates = expand_candidates(config)

    assert len(candidates) == 27


def test_candidate_id_is_stable():
    config = _config(
        [
            {
                "name": "linear_svm_pca",
                "type": "linear_svm",
                "preprocessing": {"n_components": [4]},
                "params": {"C": [1.0]},
            }
        ]
    )

    first = expand_candidates(config)[0]
    second = expand_candidates(config)[0]

    assert first.candidate_id == second.candidate_id


def test_candidate_id_changes_when_params_change():
    base = _config(
        [
            {
                "name": "linear_svm_pca",
                "type": "linear_svm",
                "preprocessing": {"n_components": [4]},
                "params": {"C": [1.0]},
            }
        ]
    )
    changed = _config(
        [
            {
                "name": "linear_svm_pca",
                "type": "linear_svm",
                "preprocessing": {"n_components": [4]},
                "params": {"C": [10.0]},
            }
        ]
    )

    assert expand_candidates(base)[0].candidate_id != expand_candidates(changed)[
        0
    ].candidate_id


def test_candidate_expansion_rejects_empty_grid():
    config = _config(
        [
            {
                "name": "linear_svm_pca",
                "type": "linear_svm",
                "preprocessing": {"n_components": []},
                "params": {"C": [1.0]},
            }
        ]
    )

    try:
        expand_candidates(config)
    except ValueError as exc:
        assert "empty grid" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_angle_range_is_treated_as_scalar_value():
    config = _config(
        [
            {
                "name": "linear_svm_pca",
                "type": "linear_svm",
                "preprocessing": {
                    "n_components": [4, 6],
                    "angle_range": [-3.14, 3.14],
                },
                "params": {"C": [1.0]},
            }
        ]
    )

    candidates = expand_candidates(config)

    assert len(candidates) == 2
    assert candidates[0].preprocessing["angle_range"] == [-3.14, 3.14]


def test_classical_baselines_candidate_count_is_reasonable():
    config = load_config("configs/classical_baselines.yaml")

    candidates = expand_candidates(config)

    assert len(candidates) == 76


def test_quantum_candidate_expansion_includes_encoding_and_depth():
    config = load_config("configs/dev_smoke.yaml")

    candidate = expand_candidates(config)[0]

    assert candidate.model_type == "quantum_kernel_svm"
    assert candidate.quantum == {"encoding": "angle_y", "depth": 1, "alpha": 1.0}

