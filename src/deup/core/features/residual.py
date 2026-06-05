"""kNN-smoothed residual magnitude feature."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted


class ResidualMagnitude(BaseEstimator, TransformerMixin):
    """Local average of training residual magnitudes |y - f(x)|.

    At ``fit(X, y)`` the base ``estimator`` is cloned and fit, and absolute residuals
    are stored. At ``transform(X)`` (inference, no ``y``), returns the mean residual
    magnitude among ``k`` nearest training neighbors — a local error prior.

    Parameters
    ----------
    estimator:
        Base model ``f`` (any sklearn ``fit``/``predict`` object). Cloned at fit.
    k:
        Neighbors for smoothing.
    """

    def __init__(self, estimator: Any, *, k: int = 5) -> None:
        self.estimator = estimator
        self.k = k

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> ResidualMagnitude:
        if y is None:
            raise ValueError("ResidualMagnitude requires y at fit.")
        X_arr = check_array(X, accept_sparse=False)
        y_arr = np.asarray(y, dtype=float)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_arr, y_arr)
        pred = np.asarray(self.estimator_.predict(X_arr), dtype=float)
        self.residuals_ = np.abs(y_arr - pred)
        self.X_train_ = X_arr
        self.n_features_in_ = X_arr.shape[1]
        k_eff = min(self.k, len(X_arr))
        self._nn = NearestNeighbors(n_neighbors=k_eff)
        self._nn.fit(X_arr)
        self._k_eff = k_eff
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "residuals_")
        X_arr = check_array(X, accept_sparse=False)
        _, idx = self._nn.kneighbors(X_arr, n_neighbors=self._k_eff)
        neighbor_res = self.residuals_[idx]
        smoothed = neighbor_res.mean(axis=1)
        return np.asarray(smoothed.reshape(-1, 1), dtype=float)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[Any]:
        del input_features
        return np.array(["residual_magnitude"], dtype=object)
