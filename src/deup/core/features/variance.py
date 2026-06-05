"""Predictive-variance features log V(x) for g(x)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted

VarianceMethod = Literal["ensemble", "gp"]


class VarianceFeature(BaseEstimator, TransformerMixin):
    """Estimate log predictive variance log V(x).

    Parameters
    ----------
    method:
        ``"ensemble"`` — bootstrap variance of a cloned base estimator's predictions.
        ``"gp"`` — GP variance (requires ``pip install "deup[torch]"``; not yet
        implemented — raises at fit).
    estimator:
        Base model for ``method="ensemble"`` (cloned per bootstrap replicate).
    n_estimators:
        Number of bootstrap replicas.
    random_state:
        Seed for bootstrap subsampling.
    eps:
        Added inside ``log(var + eps)`` for numerical stability.
    """

    def __init__(
        self,
        method: VarianceMethod = "ensemble",
        estimator: Any = None,
        *,
        n_estimators: int = 10,
        random_state: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.method = method
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.eps = eps

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> VarianceFeature:
        if self.method == "gp":
            _require_torch("VarianceFeature(method='gp')")
            raise NotImplementedError(
                "GP variance (DUE-style) is reserved for a future gpytorch release."
            )
        if y is None:
            raise ValueError("VarianceFeature(method='ensemble') requires y at fit.")
        X_arr = check_array(X, accept_sparse=False)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("y and X must have the same number of rows")

        from sklearn.ensemble import HistGradientBoostingRegressor

        base = self.estimator if self.estimator is not None else HistGradientBoostingRegressor()
        self._models: list[Any] = []
        rng = np.random.default_rng(self.random_state)
        n = X_arr.shape[0]
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            model = clone(base)
            model.fit(X_arr[idx], y_arr[idx])
            self._models.append(model)
        self.n_features_in_ = X_arr.shape[1]
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "n_features_in_")
        X_arr = check_array(X, accept_sparse=False)
        preds = np.stack([np.asarray(m.predict(X_arr), dtype=float) for m in self._models], axis=1)
        var = preds.var(axis=1)
        log_v = np.log(var + self.eps)
        return np.asarray(log_v.reshape(-1, 1), dtype=float)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[Any]:
        del input_features
        return np.array(["log_variance"], dtype=object)


def _require_torch(context: str) -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise ImportError(f'{context} requires torch. Install with: pip install "deup[torch]"')
