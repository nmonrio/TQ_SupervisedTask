from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted


class TabularPreprocessor:
    """Standardize, optionally reduce dimension and map features to angles."""

    def __init__(
        self,
        n_components: int | None,
        angle_range: tuple[float, float] = (-np.pi, np.pi),
        clip_angles: bool = True,
    ) -> None:
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be a positive integer or None")
        if len(angle_range) != 2 or angle_range[0] >= angle_range[1]:
            raise ValueError("angle_range must be an increasing pair")

        self.n_components = n_components
        self.angle_range = angle_range
        self.clip_angles = clip_angles
        self.standard_scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if n_components is not None else None
        self.angle_scaler = MinMaxScaler(feature_range=angle_range)

    def fit(self, X: np.ndarray) -> "TabularPreprocessor":
        X = self._validate_X(X)
        if self.n_components is not None and self.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed the number of features")

        X_scaled = self.standard_scaler.fit_transform(X)
        X_reduced = self.pca.fit_transform(X_scaled) if self.pca is not None else X_scaled
        self.angle_scaler.fit(X_reduced)
        self.n_fit_samples_ = X.shape[0]
        self.n_output_features_ = X_reduced.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["n_fit_samples_", "n_output_features_"])
        X = self._validate_X(X)
        X_scaled = self.standard_scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled) if self.pca is not None else X_scaled
        X_angles = self.angle_scaler.transform(X_reduced)
        if self.clip_angles:
            X_angles = np.clip(X_angles, self.angle_range[0], self.angle_range[1])
        return X_angles

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        return X

