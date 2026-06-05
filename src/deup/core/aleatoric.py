"""Aleatoric-uncertainty estimators a(x).

In the DEUP decomposition the epistemic estimate is ``e_hat(x) = max(0, g(x) - a(x))``
where ``a(x)`` estimates the irreducible (aleatoric) risk ``A(x) = R(f*, x)`` — the
loss the Bayes predictor still incurs because of label noise (Lahlou et al., 2023,
Sec. 3; Eq. 7).

Under squared-error loss ``A(x) = Var(Y | X=x)`` (Example 1 in the paper), so these
estimators target the conditional variance of ``y``. They are deliberately
**model-agnostic**: each estimates label-noise spread directly from ``(X, y)`` without
reference to the base model ``f``, so the subtraction ``g - a`` removes noise rather
than the model's own error.

Estimators
----------
- :class:`Homoscedastic` — a single global variance ``a(x) = sigma^2``.
- :class:`Heteroscedastic` — local label variance via k-NN (input-dependent).
- :class:`Quantile` — spread from quantile regression, ``a(x) = ((q_hi - q_lo)/z)^2``.

All return **variance-scale** estimates (same units as a squared-error target) and are
clipped to be non-negative.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted


class Homoscedastic(BaseEstimator):
    """Constant aleatoric variance ``a(x) = sigma^2`` for all ``x``.

    The global noise level is estimated as the mean local label variance among
    ``k`` nearest neighbors (a bias-corrected estimate of ``Var(Y | X)`` averaged over
    the training inputs). Use when label noise is believed roughly constant across the
    input space (the paper's scenario 3 with a non-zero floor).

    Parameters
    ----------
    k:
        Neighbors used to estimate local label variance.
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Homoscedastic:
        X_arr = check_array(X, accept_sparse=False)
        y_arr = np.asarray(y, dtype=float)
        local = _knn_local_variance(X_arr, y_arr, k=self.k)
        self.sigma2_ = float(np.mean(local))
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "sigma2_")
        n = check_array(X, accept_sparse=False).shape[0]
        return np.full(n, self.sigma2_, dtype=float)


class Heteroscedastic(BaseEstimator):
    """Input-dependent aleatoric variance via local k-NN label variance.

    For each ``x`` the estimate is the bias-corrected variance of training ``y`` among
    its ``k`` nearest neighbors — a model-free estimate of ``Var(Y | X = x)``.

    Parameters
    ----------
    k:
        Number of neighbors for the local variance estimate.
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Heteroscedastic:
        X_arr = check_array(X, accept_sparse=False)
        self._y = np.asarray(y, dtype=float)
        k_eff = min(self.k, len(X_arr))
        self._k_eff = k_eff
        self._nn = NearestNeighbors(n_neighbors=k_eff)
        self._nn.fit(X_arr)
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "_nn")
        X_arr = check_array(X, accept_sparse=False)
        _, idx = self._nn.kneighbors(X_arr, n_neighbors=self._k_eff)
        neighbor_y = self._y[idx]
        var = neighbor_y.var(axis=1, ddof=1) if self._k_eff > 1 else np.zeros(len(X_arr))
        return np.clip(np.asarray(var, dtype=float), 0.0, None)


class Quantile(BaseEstimator):
    """Aleatoric variance from a quantile-regression spread.

    Fits two quantile regressors at ``q_lo`` and ``q_hi`` and converts the predicted
    interval width to a variance via the Gaussian relation
    ``sigma = (q_hi - q_lo) / (z_hi - z_lo)``, then ``a(x) = sigma^2``.

    Parameters
    ----------
    estimator:
        A quantile regressor factory taking a ``quantile=`` kwarg. Defaults to
        :class:`~sklearn.ensemble.HistGradientBoostingRegressor` with
        ``loss="quantile"``.
    q_lo, q_hi:
        Lower / upper quantiles (default 0.159 / 0.841 ~ +/-1 sigma).
    """

    def __init__(
        self,
        estimator: Any = None,
        *,
        q_lo: float = 0.159,
        q_hi: float = 0.841,
    ) -> None:
        self.estimator = estimator
        self.q_lo = q_lo
        self.q_hi = q_hi

    def _make(self, quantile: float) -> Any:
        if self.estimator is not None:
            est = clone(self.estimator)
            est.set_params(quantile=quantile)
            return est
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Quantile:
        X_arr = check_array(X, accept_sparse=False)
        y_arr = np.asarray(y, dtype=float)
        self.lo_ = self._make(self.q_lo).fit(X_arr, y_arr)
        self.hi_ = self._make(self.q_hi).fit(X_arr, y_arr)
        self._z = float(norm.ppf(self.q_hi) - norm.ppf(self.q_lo))
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "lo_")
        X_arr = check_array(X, accept_sparse=False)
        lo = np.asarray(self.lo_.predict(X_arr), dtype=float)
        hi = np.asarray(self.hi_.predict(X_arr), dtype=float)
        width = np.clip(hi - lo, 0.0, None)
        sigma = width / self._z
        return np.asarray(sigma**2, dtype=float)


def _knn_local_variance(X: npt.NDArray[Any], y: npt.NDArray[Any], k: int) -> npt.NDArray[Any]:
    """Bias-corrected variance of ``y`` among each row's ``k`` nearest neighbors."""
    k_eff = min(k, len(X))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X)
    _, idx = nn.kneighbors(X, n_neighbors=k_eff)
    neighbor_y = y[idx]
    if k_eff <= 1:
        return np.zeros(len(X), dtype=float)
    return np.asarray(neighbor_y.var(axis=1, ddof=1), dtype=float)
