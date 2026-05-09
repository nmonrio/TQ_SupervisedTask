import numpy as np
import pennylane as qml
import pytest

from qml_bc.encodings import FeatureMapSpec, apply_feature_map


def test_feature_map_rejects_wrong_input_dimension():
    spec = FeatureMapSpec(name="angle_y", n_qubits=2)

    with pytest.raises(ValueError, match="n_qubits"):
        apply_feature_map(np.array([0.1]), spec)


def test_feature_map_rejects_unknown_name():
    spec = FeatureMapSpec(name="unknown", n_qubits=2)

    with pytest.raises(ValueError, match="unknown feature map"):
        apply_feature_map(np.array([0.1, 0.2]), spec)


def test_feature_map_rejects_non_positive_depth():
    with pytest.raises(ValueError, match="depth"):
        FeatureMapSpec(name="angle_y", n_qubits=2, depth=0)


def test_feature_maps_produce_statevectors():
    x = np.array([0.1, -0.2, 0.3])

    for name in ("angle_y", "angle_y_ring", "zz_ring"):
        spec = FeatureMapSpec(name=name, n_qubits=3, depth=2)
        device = qml.device("default.qubit", wires=3)

        @qml.qnode(device)
        def circuit(sample):
            apply_feature_map(sample, spec)
            return qml.state()

        state = circuit(x)

        assert state.shape == (2**3,)
        assert np.linalg.norm(state) == pytest.approx(1.0)

