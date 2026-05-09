"""Model wrappers used by the shared evaluation path."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from qml_bc.quantum_kernel import QuantumStateKernel


class ClassifierProtocol(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassifierProtocol": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def decision_function(self, X: np.ndarray) -> np.ndarray: ...


class QuantumKernelSVC:
    """Support-vector classifier using a precomputed quantum fidelity kernel."""

    def __init__(
        self,
        kernel: QuantumStateKernel,
        C: float = 1.0,
    ) -> None:
        self.kernel = kernel
        self.C = C
        self._svc = SVC(kernel="precomputed", C=C)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelSVC":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        K_train = self.kernel.matrix(X)
        self._svc.fit(K_train, y)
        self.X_train_ = X.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["X_train_"])
        K_test = self.kernel.matrix(np.asarray(X, dtype=float), self.X_train_)
        return self._svc.predict(K_test)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["X_train_"])
        K_test = self.kernel.matrix(np.asarray(X, dtype=float), self.X_train_)
        return self._svc.decision_function(K_test)

