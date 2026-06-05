"""Input-space density features log q(x | z^N)."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted

DensityMethod = Literal["mahalanobis", "knn", "kde", "flow"]


def _diagonal_gaussian_log_prob(
    X: npt.NDArray[Any],
    mu: npt.NDArray[Any],
    log_sigma2: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Closed-form diagonal-Gaussian log-density (thesis ``GaussianDensity.log_prob``).

    log p(x) = -0.5 * sum_d [(x_d - mu_d)^2 / sigma2_d + log sigma2_d]
               - D/2 * log(2*pi)
    """
    sigma2 = np.exp(log_sigma2)
    diff2 = (X - mu) ** 2 / sigma2
    d = X.shape[1]
    return np.asarray(
        -0.5 * (diff2 + log_sigma2).sum(axis=1) - 0.5 * d * math.log(2 * math.pi),
        dtype=float,
    )


class DensityFeature(BaseEstimator, TransformerMixin):
    """Estimate log-density log q(x | training data).

    Parameters
    ----------
    method:
        ``"mahalanobis"`` — diagonal Gaussian MLE (Lee et al. 2018 / thesis default).
        ``"knn"`` — ``-log(d_k + eps)`` where ``d_k`` is the k-th NN distance.
        ``"kde"`` — sklearn ``KernelDensity`` with Gaussian kernel.
        ``"flow"`` — normalizing flow (requires ``pip install "deup[torch]"``).
    k:
        Neighbors for ``method="knn"``.
    bandwidth:
        Bandwidth for ``method="kde"``.
    var_floor:
        Minimum variance per dimension for ``mahalanobis`` (dead-ReLU dimensions).
    eps:
        Stabilizer for ``knn`` log-density.
    """

    def __init__(
        self,
        method: DensityMethod = "mahalanobis",
        *,
        k: int = 5,
        bandwidth: float = 1.0,
        var_floor: float = 1e-6,
        eps: float = 1e-8,
    ) -> None:
        self.method = method
        self.k = k
        self.bandwidth = bandwidth
        self.var_floor = var_floor
        self.eps = eps

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> DensityFeature:
        del y
        if self.method == "flow":
            _require_torch("DensityFeature(method='flow')")
        X_arr = check_array(X, accept_sparse=False)
        self._X_train = X_arr
        self.n_features_in_ = X_arr.shape[1]

        if self.method == "mahalanobis":
            self.mu_ = X_arr.mean(axis=0)
            sigma2 = X_arr.var(axis=0)
            self.log_sigma2_ = np.log(np.maximum(sigma2, self.var_floor))
        elif self.method == "knn":
            self._nn = NearestNeighbors(n_neighbors=min(self.k, len(X_arr)))
            self._nn.fit(X_arr)
        elif self.method == "kde":
            self._kde = KernelDensity(bandwidth=self.bandwidth, kernel="gaussian")
            self._kde.fit(X_arr)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "n_features_in_")
        X_arr = check_array(X, accept_sparse=False)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")

        if self.method == "mahalanobis":
            log_q = _diagonal_gaussian_log_prob(X_arr, self.mu_, self.log_sigma2_)
        elif self.method == "knn":
            k_eff = min(self.k, len(self._X_train))
            dists, _ = self._nn.kneighbors(X_arr, n_neighbors=k_eff)
            d_k = dists[:, -1]
            log_q = -np.log(d_k + self.eps)
        elif self.method == "kde":
            log_q = self._kde.score_samples(X_arr)
        else:  # flow — fit not implemented in v0.1; guarded at __init__ fit
            raise NotImplementedError(
                "Normalizing-flow density is reserved for a future torch release."
            )
        return np.asarray(log_q.reshape(-1, 1), dtype=float)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[Any]:
        del input_features
        return np.array(["log_density"], dtype=object)


def _require_torch(context: str) -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise ImportError(
            f'{context} requires torch. Install with: pip install "deup[torch]"'
        )
