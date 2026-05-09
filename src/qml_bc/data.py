"""Dataset loading utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer


@dataclass(frozen=True)
class DatasetBundle:
    """In-memory representation of a supervised tabular dataset."""

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    positive_label: str


def load_breast_cancer_dataset(
    positive_label: str = "malignant",
) -> DatasetBundle:
    """Load Breast Cancer Wisconsin with an explicit positive-class convention.

    Scikit-learn stores the original labels as 0 = malignant and 1 = benign.
    This project remaps the selected positive label to integer 1 so metrics such
    as positive recall have an unambiguous medical interpretation.
    """

    raw = load_breast_cancer()
    target_names = tuple(str(name) for name in raw.target_names)

    if positive_label not in target_names:
        valid = ", ".join(target_names)
        raise ValueError(f"positive_label must be one of: {valid}")

    positive_idx = target_names.index(positive_label)
    y = (raw.target == positive_idx).astype(int)

    return DatasetBundle(
        X=np.asarray(raw.data, dtype=float),
        y=y,
        feature_names=tuple(str(name) for name in raw.feature_names),
        target_names=target_names,
        positive_label=positive_label,
    )

