"""Shared experiment evaluation for classical and quantum-kernel models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qml_bc.candidates import CandidateSpec, expand_candidates
from qml_bc.config import ExperimentConfig
from qml_bc.data import load_breast_cancer_dataset
from qml_bc.encodings import FeatureMapSpec
from qml_bc.metrics import classification_metrics
from qml_bc.models import ClassifierProtocol, QuantumKernelSVC, build_classical_model
from qml_bc.preprocessing import TabularPreprocessor
from qml_bc.quantum_kernel import QuantumStateKernel, StatevectorFeatureMap
from qml_bc.splitting import SplitIndices, make_inner_splits, make_outer_splits


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Run nested evaluation and return one result row per outer split."""

    data = load_breast_cancer_dataset(
        positive_label=config.dataset.get("positive_label", "malignant")
    )
    candidates = expand_candidates(config)
    rows: list[dict[str, Any]] = []

    for outer_split in make_outer_splits(data.y, config.splitting):
        X_train = data.X[outer_split.train_idx]
        y_train = data.y[outer_split.train_idx]
        X_test = data.X[outer_split.test_idx]
        y_test = data.y[outer_split.test_idx]

        selected = select_candidate_with_inner_cv(
            candidates=candidates,
            X_train=X_train,
            y_train=y_train,
            inner_folds=config.splitting.inner_folds,
            seed=outer_split.seed,
        )

        fitted = fit_candidate(selected.candidate, X_train, y_train)
        y_pred = fitted.model.predict(fitted.preprocessor.transform(X_test))
        y_score = fitted.model.decision_function(fitted.preprocessor.transform(X_test))
        metrics = classification_metrics(y_test, y_pred, y_score)

        rows.append(
            {
                "experiment_name": config.experiment_name,
                "split_id": outer_split.split_id,
                "seed": outer_split.seed,
                "model_name": selected.candidate.model_name,
                "candidate_id": selected.candidate.candidate_id,
                "model_type": selected.candidate.model_type,
                "n_components": selected.candidate.preprocessing.get("n_components"),
                "encoding_name": _quantum_value(selected.candidate, "encoding"),
                "depth": _quantum_value(selected.candidate, "depth"),
                "C": selected.candidate.model_params.get("C"),
                "gamma": selected.candidate.model_params.get("gamma"),
                "inner_cv_score": selected.inner_cv_score,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


class FittedCandidate:
    def __init__(
        self,
        preprocessor: TabularPreprocessor,
        model: ClassifierProtocol,
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model


class SelectedCandidate:
    def __init__(self, candidate: CandidateSpec, inner_cv_score: float) -> None:
        self.candidate = candidate
        self.inner_cv_score = inner_cv_score


def select_candidate_with_inner_cv(
    candidates: list[CandidateSpec],
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_folds: int,
    seed: int,
) -> SelectedCandidate:
    """Select the candidate with best mean inner balanced accuracy."""

    folds = make_inner_splits(y_train, n_folds=inner_folds, seed=seed)
    best_candidate: CandidateSpec | None = None
    best_score = -np.inf

    for candidate in candidates:
        scores = []
        for inner_train_idx, valid_idx in folds:
            fitted = fit_candidate(
                candidate,
                X_train[inner_train_idx],
                y_train[inner_train_idx],
            )
            X_valid = fitted.preprocessor.transform(X_train[valid_idx])
            y_pred = fitted.model.predict(X_valid)
            y_score = fitted.model.decision_function(X_valid)
            scores.append(
                classification_metrics(y_train[valid_idx], y_pred, y_score)[
                    "balanced_accuracy"
                ]
            )

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_candidate = candidate
            best_score = mean_score

    if best_candidate is None:
        raise ValueError("at least one candidate is required")

    return SelectedCandidate(candidate=best_candidate, inner_cv_score=best_score)


def fit_candidate(
    candidate: CandidateSpec,
    X: np.ndarray,
    y: np.ndarray,
) -> FittedCandidate:
    """Fit preprocessing and model for one candidate without data leakage."""

    preprocessor = _build_preprocessor(candidate.preprocessing)
    X_preprocessed = preprocessor.fit_transform(X)
    model = build_model_for_candidate(candidate)
    model.fit(X_preprocessed, y)
    return FittedCandidate(preprocessor=preprocessor, model=model)


def build_model_for_candidate(candidate: CandidateSpec) -> ClassifierProtocol:
    if candidate.model_type in {"linear_svm", "rbf_svm"}:
        return build_classical_model(candidate.model_type, candidate.model_params)
    if candidate.model_type == "quantum_kernel_svm":
        if candidate.quantum is None:
            raise ValueError("quantum_kernel_svm requires quantum config")
        n_qubits = candidate.preprocessing.get("n_components")
        if not isinstance(n_qubits, int):
            raise ValueError("quantum_kernel_svm requires integer n_components")
        spec = FeatureMapSpec(
            name=str(candidate.quantum["encoding"]),
            n_qubits=n_qubits,
            depth=int(candidate.quantum.get("depth", 1)),
            alpha=float(candidate.quantum.get("alpha", 1.0)),
        )
        kernel = QuantumStateKernel(StatevectorFeatureMap(spec))
        return QuantumKernelSVC(kernel=kernel, C=float(candidate.model_params.get("C", 1.0)))
    raise ValueError(f"unknown model type: {candidate.model_type}")


def _build_preprocessor(preprocessing: dict[str, Any]) -> TabularPreprocessor:
    angle_range = preprocessing.get("angle_range", (-np.pi, np.pi))
    return TabularPreprocessor(
        n_components=preprocessing.get("n_components"),
        angle_range=(float(angle_range[0]), float(angle_range[1])),
    )


def _quantum_value(candidate: CandidateSpec, key: str) -> Any:
    if candidate.quantum is None:
        return None
    return candidate.quantum.get(key)

