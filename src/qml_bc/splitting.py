from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from qml_bc.config import SplitConfig


@dataclass(frozen=True)
class SplitIndices:
    split_id: int
    seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_outer_splits(y: np.ndarray, config: SplitConfig) -> list[SplitIndices]:
    """Create repeated stratified train/test splits from seeds."""

    y = np.asarray(y)
    indices = np.arange(len(y))
    splits: list[SplitIndices] = []

    for split_id, seed in enumerate(config.seeds[: config.outer_splits]):
        train_idx, test_idx = train_test_split(
            indices,
            test_size=config.test_size,
            random_state=seed,
            stratify=y,
        )
        splits.append(
            SplitIndices(
                split_id=split_id,
                seed=seed,
                train_idx=np.asarray(train_idx, dtype=int),
                test_idx=np.asarray(test_idx, dtype=int),
            )
        )

    return splits


def make_inner_splits(
    y_train: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified folds over the already-isolated training labels."""

    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    y_train = np.asarray(y_train)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (np.asarray(train_idx, dtype=int), np.asarray(valid_idx, dtype=int))
        for train_idx, valid_idx in splitter.split(np.zeros_like(y_train), y_train)
    ]

