"""Distance-to-training-set features."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted


class DistanceToTrain(BaseEstimator, TransformerMixin):
    """k-th nearest-neighbor distance from ``x`` to the training manifold.

    Parameters
    ----------
    k:
        Which neighbor distance to return (1 = nearest). Useful as a covariate-shift
        proxy complementary to ``DensityFeature``.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> DistanceToTrain:
        del y
        X_arr = check_array(X, accept_sparse=False)
        self.n_features_in_ = X_arr.shape[1]
        k_eff = min(self.k, len(X_arr))
        self._k_eff = k_eff
        self._nn = NearestNeighbors(n_neighbors=k_eff)
        self._nn.fit(X_arr)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "n_features_in_")
        X_arr = check_array(X, accept_sparse=False)
        dists, _ = self._nn.kneighbors(X_arr, n_neighbors=self._k_eff)
        return np.asarray(dists[:, -1].reshape(-1, 1), dtype=float)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[Any]:
        del input_features
        return np.array([f"dist_train_k{self.k}"], dtype=object)
