import numpy as np

from qml_bc.config import SplitConfig
from qml_bc.data import load_breast_cancer_dataset
from qml_bc.splitting import make_inner_splits, make_outer_splits


def test_outer_splits_are_deterministic():
    y = load_breast_cancer_dataset().y
    config = SplitConfig(outer_splits=2, test_size=0.2, inner_folds=2, seeds=(3, 4))

    first = make_outer_splits(y, config)
    second = make_outer_splits(y, config)

    assert np.array_equal(first[0].train_idx, second[0].train_idx)
    assert np.array_equal(first[0].test_idx, second[0].test_idx)


def test_train_test_do_not_overlap():
    y = load_breast_cancer_dataset().y
    config = SplitConfig(outer_splits=1, test_size=0.2, inner_folds=2, seeds=(0,))
    split = make_outer_splits(y, config)[0]

    assert set(split.train_idx).isdisjoint(set(split.test_idx))


def test_outer_splits_preserve_class_balance_approximately():
    y = load_breast_cancer_dataset().y
    config = SplitConfig(outer_splits=1, test_size=0.2, inner_folds=2, seeds=(0,))
    split = make_outer_splits(y, config)[0]

    overall_rate = y.mean()
    train_rate = y[split.train_idx].mean()
    test_rate = y[split.test_idx].mean()

    assert abs(train_rate - overall_rate) < 0.02
    assert abs(test_rate - overall_rate) < 0.03


def test_inner_splits_do_not_overlap():
    y = load_breast_cancer_dataset().y[:100]
    splits = make_inner_splits(y, n_folds=3, seed=0)

    for train_idx, valid_idx in splits:
        assert set(train_idx).isdisjoint(set(valid_idx))

