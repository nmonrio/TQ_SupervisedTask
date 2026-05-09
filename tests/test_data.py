import numpy as np
import pytest

from qml_bc.data import load_breast_cancer_dataset


def test_load_dataset_shapes():
    bundle = load_breast_cancer_dataset()

    assert bundle.X.shape == (569, 30)
    assert bundle.y.shape == (569,)
    assert len(bundle.feature_names) == 30


def test_labels_are_binary():
    bundle = load_breast_cancer_dataset()

    assert set(np.unique(bundle.y)) == {0, 1}


def test_malignant_is_positive_class():
    bundle = load_breast_cancer_dataset(positive_label="malignant")

    assert bundle.positive_label == "malignant"
    assert int(bundle.y.sum()) == 212


def test_invalid_positive_label_raises():
    with pytest.raises(ValueError, match="positive_label"):
        load_breast_cancer_dataset(positive_label="unknown")

