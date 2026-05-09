"""Statevector fidelity kernels built with PennyLane."""

from __future__ import annotations

import numpy as np
import pennylane as qml

from qml_bc.encodings import FeatureMapSpec, apply_feature_map


class StatevectorFeatureMap:
    """Evaluate a feature map as a statevector on PennyLane's default simulator."""

    def __init__(self, spec: FeatureMapSpec) -> None:
        self.spec = spec
        device = qml.device("default.qubit", wires=spec.n_qubits)

        @qml.qnode(device)
        def circuit(x: np.ndarray) -> np.ndarray:
            apply_feature_map(x, spec)
            return qml.state()

        self._circuit = circuit

    def state(self, x: np.ndarray) -> np.ndarray:
        """Return the normalized statevector for one preprocessed sample."""

        x = np.asarray(x, dtype=float)
        if x.shape != (self.spec.n_qubits,):
            raise ValueError("x must have shape (n_qubits,)")
        return np.asarray(self._circuit(x), dtype=complex)


class QuantumStateKernel:
    """Fidelity kernel K(x, z) = |<psi(x)|psi(z)>|^2."""

    def __init__(self, feature_map: StatevectorFeatureMap) -> None:
        self.feature_map = feature_map

    def matrix(
        self,
        X_left: np.ndarray,
        X_right: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute a training or cross-kernel matrix."""

        X_left = self._validate_X(X_left)
        states_left = self._states(X_left)

        if X_right is None:
            states_right = states_left
        else:
            X_right = self._validate_X(X_right)
            states_right = self._states(X_right)

        K = np.abs(states_left @ states_right.conj().T) ** 2

        if X_right is None:
            K = 0.5 * (K + K.T)
            np.fill_diagonal(K, 1.0)

        return np.asarray(K, dtype=float)

    def _states(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.feature_map.state(x) for x in X])

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.feature_map.spec.n_qubits:
            raise ValueError("X must have n_qubits columns")
        return X

