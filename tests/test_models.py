import numpy as np

from qml_bc.encodings import FeatureMapSpec
from qml_bc.models import QuantumKernelSVC
from qml_bc.quantum_kernel import QuantumStateKernel, StatevectorFeatureMap


def _model() -> QuantumKernelSVC:
    spec = FeatureMapSpec(name="angle_y", n_qubits=2)
    kernel = QuantumStateKernel(StatevectorFeatureMap(spec))
    return QuantumKernelSVC(kernel=kernel, C=1.0)


def test_quantum_kernel_svc_fit_predict_shapes():
    X = np.array([[-0.8, -0.7], [-0.6, -0.5], [0.6, 0.5], [0.8, 0.7]])
    y = np.array([0, 0, 1, 1])
    model = _model()

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (4,)
    assert set(predictions).issubset({0, 1})


def test_quantum_kernel_svc_decision_function_shape():
    X_train = np.array([[-0.8, -0.7], [-0.6, -0.5], [0.6, 0.5], [0.8, 0.7]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.2, 0.1], [-0.2, -0.1]])
    model = _model().fit(X_train, y_train)

    scores = model.decision_function(X_test)

    assert scores.shape == (2,)


def test_quantum_kernel_svc_stores_training_data_for_cross_kernel():
    X = np.array([[-0.8, -0.7], [-0.6, -0.5], [0.6, 0.5], [0.8, 0.7]])
    y = np.array([0, 0, 1, 1])
    model = _model()

    model.fit(X, y)

    assert np.array_equal(model.X_train_, X)

