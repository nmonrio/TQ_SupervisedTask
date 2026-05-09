"""Quantum feature maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class FeatureMapSpec:
    name: str
    n_qubits: int
    depth: int = 1
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.depth < 1:
            raise ValueError("depth must be positive")


def apply_feature_map(x: np.ndarray, spec: FeatureMapSpec) -> None:
    """Apply the configured feature map to the active PennyLane tape."""

    x = np.asarray(x, dtype=float)
    if x.shape != (spec.n_qubits,):
        raise ValueError("x must have shape (n_qubits,)")
    if spec.name not in {"angle_y", "angle_y_ring", "zz_ring"}:
        raise ValueError(f"unknown feature map: {spec.name}")

    for _ in range(spec.depth):
        _apply_ry_layer(x)

        if spec.name == "angle_y_ring":
            _apply_cnot_ring(spec.n_qubits)
        elif spec.name == "zz_ring":
            _apply_zz_ring(x, spec.alpha)


def _apply_ry_layer(x: np.ndarray) -> None:
    for wire, angle in enumerate(x):
        qml.RY(angle, wires=wire)


def _apply_cnot_ring(n_qubits: int) -> None:
    if n_qubits == 1:
        return
    for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
    qml.CNOT(wires=[n_qubits - 1, 0])


def _apply_zz_ring(x: np.ndarray, alpha: float) -> None:
    n_qubits = len(x)
    if n_qubits == 1:
        qml.RZ(alpha * x[0] * x[0], wires=0)
        return

    for left in range(n_qubits):
        right = (left + 1) % n_qubits
        qml.CNOT(wires=[left, right])
        qml.RZ(alpha * x[left] * x[right], wires=right)
        qml.CNOT(wires=[left, right])

