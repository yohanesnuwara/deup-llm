"""The DEUP error predictor g(x) as a reusable component.

``ErrorEstimator`` fits a secondary model on ``(features(X), oof_errors)`` — the
out-of-fold pointwise losses produced by :class:`~deup.core.oof.OOFErrorCollector`.
It encapsulates the three pieces of the DEUP error model that were previously inlined
in ``DEUPRegressor``:

1. an optional **feature pipeline** (the stationarizing features phi_{z^N}(x) from
   Lahlou et al., 2023, Sec. 3.2 — see :mod:`deup.core.features`),
2. an **error-target transform** (``log`` / ``asinh`` / ``none``) that stabilizes the
   heavy-tailed regression target, and
3. **non-negativity** of the returned error estimate (losses are non-negative).

It is the building block ``DEUPRanker`` / ``DEUPClassifier`` (P7) compose on top of.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted

from deup.core.losses import (
    TargetTransform,
    apply_error_transform,
    inverse_error_transform,
)


class ErrorEstimator(BaseEstimator):
    """Fit and predict the DEUP error model ``g``.

    Parameters
    ----------
    model:
        The secondary regressor. Defaults to
        :class:`~sklearn.ensemble.HistGradientBoostingRegressor` (no extra deps).
        Pass any sklearn-style regressor, or use ``"lightgbm"`` to request a
        LightGBM model (requires the ``[gbm]`` extra).
    features:
        Optional feature builder (e.g. a
        :class:`~deup.core.features.FeaturePipeline`) applied to ``X`` before fitting
        ``model``. If ``None``, raw ``X`` is used.
    target_transform:
        Stabilization for the error target: ``"log"`` (default), ``"asinh"`` or
        ``"none"``.
    error_eps:
        Stabilizer for ``log`` / ``asinh`` transforms.
    clip_negative:
        If ``True`` (default), clip predicted errors at 0.

    Attributes
    ----------
    model_ :
        The fitted secondary regressor.
    features_ :
        The fitted feature builder (or ``None``).
    """

    def __init__(
        self,
        model: Any = None,
        features: Any = None,
        *,
        target_transform: TargetTransform = "log",
        error_eps: float = 1e-6,
        clip_negative: bool = True,
    ) -> None:
        self.model = model
        self.features = features
        self.target_transform = target_transform
        self.error_eps = error_eps
        self.clip_negative = clip_negative

    def _resolve_model(self) -> Any:
        if self.model is None:
            return HistGradientBoostingRegressor()
        if isinstance(self.model, str):
            key = self.model
            if key == "lgbm":
                key = "lightgbm"
            if key == "lightgbm":
                try:
                    from lightgbm import LGBMRegressor
                except ImportError as exc:
                    raise ImportError('model="lightgbm" requires: pip install "deup[gbm]"') from exc
                return LGBMRegressor(n_estimators=100, verbose=-1)
            if key == "xgb":
                try:
                    from xgboost import XGBRegressor
                except ImportError as exc:
                    raise ImportError('model="xgb" requires: pip install "deup[xgb]"') from exc
                return XGBRegressor(n_estimators=100, verbosity=0)
            if key == "catboost":
                try:
                    from catboost import CatBoostRegressor
                except ImportError as exc:
                    raise ImportError(
                        'model="catboost" requires: pip install "deup[catboost]"'
                    ) from exc
                return CatBoostRegressor(
                    iterations=100, verbose=False, allow_writing_files=False
                )
            raise ValueError(
                f"Unknown model string: {self.model!r}. "
                "Use 'lightgbm'/'lgbm', 'xgb', or 'catboost'."
            )
        return clone(self.model)

    def _build_features(self, X: Any, y: npt.ArrayLike | None, *, fit: bool) -> Any:
        if self.features is None:
            return X
        if fit:
            self.features_ = clone(self.features)
            return self.features_.fit_transform(X, y)
        check_is_fitted(self, "model_")
        return self.features_.transform(X)

    def fit(
        self,
        X: Any,
        errors: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
    ) -> ErrorEstimator:
        """Fit ``g`` on ``(features(X), errors)``.

        Parameters
        ----------
        X:
            Inputs aligned with ``errors`` (the rows that received OOF errors).
        errors:
            Non-negative pointwise error targets from the OOF collector.
        y:
            Optional original targets, forwarded to feature builders that need them
            (e.g. ``VarianceFeature`` / ``ResidualMagnitude``).
        """
        err = np.asarray(errors, dtype=float)
        if np.any(err < 0):
            raise ValueError("error targets must be non-negative")
        feats = self._build_features(X, y, fit=True)
        target = apply_error_transform(err, method=self.target_transform, eps=self.error_eps)
        self.model_ = self._resolve_model()
        self.model_.fit(feats, target)
        return self

    def predict(self, X: Any) -> npt.NDArray[Any]:
        """Predict the (non-negative) error estimate ``g(x)``."""
        check_is_fitted(self, "model_")
        feats = self._build_features(X, None, fit=False)
        raw = np.asarray(self.model_.predict(feats), dtype=float)
        unc = inverse_error_transform(raw, method=self.target_transform, eps=self.error_eps)
        if self.clip_negative:
            unc = np.clip(unc, 0.0, None)
        return np.asarray(unc, dtype=float)
