import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from qml_bc.data import load_breast_cancer_dataset
from qml_bc.preprocessing import TabularPreprocessor


def test_transform_before_fit_raises():
    preprocessor = TabularPreprocessor(n_components=4)

    with pytest.raises(NotFittedError):
        preprocessor.transform(np.ones((3, 30)))


def test_preprocessor_reduces_dimension():
    X = load_breast_cancer_dataset().X[:50]
    preprocessor = TabularPreprocessor(n_components=4)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape == (50, 4)


def test_preprocessor_outputs_finite_values():
    X = load_breast_cancer_dataset().X[:50]
    transformed = TabularPreprocessor(n_components=4).fit_transform(X)

    assert np.isfinite(transformed).all()


def test_preprocessor_clips_to_angle_range():
    X = load_breast_cancer_dataset().X[:50]
    preprocessor = TabularPreprocessor(n_components=4)

    transformed = preprocessor.fit_transform(X)

    assert transformed.min() >= -np.pi
    assert transformed.max() <= np.pi


def test_invalid_n_components_raises():
    with pytest.raises(ValueError, match="n_components"):
        TabularPreprocessor(n_components=0)


def test_preprocessor_records_fit_sample_count():
    X = load_breast_cancer_dataset().X[:37]
    preprocessor = TabularPreprocessor(n_components=4)

    preprocessor.fit(X)

    assert preprocessor.n_fit_samples_ == 37

