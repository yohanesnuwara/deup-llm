"""Passthrough raw input features x."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class RawFeatures(BaseEstimator, TransformerMixin):
    """Return ``X`` unchanged (the ``x`` component of phi_{z^N}(x))."""

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> RawFeatures:
        self.n_features_in_ = check_array(X, accept_sparse=False).shape[1]
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "n_features_in_")
        arr = check_array(X, accept_sparse=False)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {arr.shape[1]} features, expected {self.n_features_in_}")
        return np.asarray(arr, dtype=float)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[Any]:
        check_is_fitted(self, "n_features_in_")
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
            if names.shape[0] != self.n_features_in_:
                raise ValueError("input_features length mismatch")
            return names
        return np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
