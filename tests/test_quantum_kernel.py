import numpy as np
import pytest

from qml_bc.encodings import FeatureMapSpec
from qml_bc.quantum_kernel import QuantumStateKernel, StatevectorFeatureMap


def _small_kernel(name: str = "zz_ring") -> QuantumStateKernel:
    spec = FeatureMapSpec(name=name, n_qubits=3, depth=1)
    return QuantumStateKernel(StatevectorFeatureMap(spec))


def test_statevector_has_expected_shape():
    feature_map = StatevectorFeatureMap(FeatureMapSpec(name="angle_y", n_qubits=3))

    state = feature_map.state(np.array([0.1, 0.2, 0.3]))

    assert state.shape == (2**3,)


def test_statevector_has_unit_norm():
    feature_map = StatevectorFeatureMap(FeatureMapSpec(name="zz_ring", n_qubits=3))

    state = feature_map.state(np.array([0.1, 0.2, 0.3]))

    assert np.linalg.norm(state) == pytest.approx(1.0)


def test_statevector_is_deterministic():
    feature_map = StatevectorFeatureMap(FeatureMapSpec(name="angle_y_ring", n_qubits=3))
    x = np.array([0.1, 0.2, 0.3])

    first = feature_map.state(x)
    second = feature_map.state(x)

    assert np.allclose(first, second)


def test_quantum_kernel_shape():
    X = np.array([[0.1, 0.2, 0.3], [0.3, -0.2, 0.4]])

    K = _small_kernel().matrix(X)

    assert K.shape == (2, 2)


def test_quantum_kernel_is_symmetric():
    X = np.array([[0.1, 0.2, 0.3], [0.3, -0.2, 0.4], [-0.4, 0.1, 0.2]])

    K = _small_kernel().matrix(X)

    assert np.allclose(K, K.T)


def test_quantum_kernel_has_unit_diagonal():
    X = np.array([[0.1, 0.2, 0.3], [0.3, -0.2, 0.4], [-0.4, 0.1, 0.2]])

    K = _small_kernel().matrix(X)

    assert np.allclose(np.diag(K), 1.0)


def test_quantum_kernel_is_psd_with_tolerance():
    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.3, -0.2, 0.4],
            [-0.4, 0.1, 0.2],
            [0.0, -0.3, 0.5],
        ]
    )

    K = _small_kernel().matrix(X)
    eigenvalues = np.linalg.eigvalsh(K)

    assert eigenvalues.min() >= -1e-10


def test_quantum_cross_kernel_shape():
    X_left = np.array([[0.1, 0.2, 0.3], [0.3, -0.2, 0.4]])
    X_right = np.array([[-0.4, 0.1, 0.2], [0.0, -0.3, 0.5], [0.2, 0.2, 0.2]])

    K = _small_kernel().matrix(X_left, X_right)

    assert K.shape == (2, 3)

