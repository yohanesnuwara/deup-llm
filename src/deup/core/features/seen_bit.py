"""Seen-bit feature s in phi_{z^N}(x)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class SeenBit(BaseEstimator, TransformerMixin):
    """Binary indicator: 1 if ``x`` was in the training set, else 0.

    Parameters
    ----------
    atol, rtol:
        Tolerance for floating-point duplicate detection (exact match when both 0).
    """

    def __init__(self, *, atol: float = 0.0, rtol: float = 0.0) -> None:
        self.atol = atol
        self.rtol = rtol

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> SeenBit:
        del y
        self.X_train_ = check_array(X, accept_sparse=False)
        self.n_features_in_ = self.X_train_.shape[1]
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "X_train_")
        X_arr = check_array(X, accept_sparse=False)
        n = X_arr.shape[0]
        seen = np.zeros(n, dtype=float)
        if self.atol == 0.0 and self.rtol == 0.0:
            # Exact row match via structured view (fast for moderate n).
            train_view = self.X_train_.view([("", self.X_train_.dtype)] * self.X_train_.shape[1])
            test_view = X_arr.view([("", X_arr.dtype)] * X_arr.shape[1])
            for i, row in enumerate(test_view):
                seen[i] = float(np.any(train_view == row))
        else:
            diff = X_arr[:, None, :] - self.X_train_[None, :, :]
            close = np.all(np.isclose(diff, 0.0, rtol=self.rtol, atol=self.atol), axis=2)
            seen = close.any(axis=1).astype(float)
        return seen.reshape(-1, 1)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[Any]:
        del input_features
        return np.array(["seen_bit"], dtype=object)
