"""User-facing DEUP estimators.

``DEUPRegressor`` is the ergonomic, scikit-learn-compatible entry point: wrap any
regressor, fit, and get a point prediction plus an epistemic-uncertainty estimate.

    from sklearn.ensemble import RandomForestRegressor
    from deup import DEUPRegressor

    model = DEUPRegressor(base_model=RandomForestRegressor())
    model.fit(X_train, y_train)
    pred, unc = model.predict(X_test, return_uncertainty=True)

Under the hood it composes the leakage-correct :class:`~deup.core.oof.OOFErrorCollector`
(out-of-sample errors of the base model) with a secondary "error predictor" ``g`` that
regresses those errors -- this is DEUP (Lahlou et al., 2023). In this minimal v0.1
the aleatoric term is taken as zero, so the reported epistemic uncertainty is the
predicted out-of-sample error ``g(x)`` (the paper's conservative proxy); the
aleatoric decomposition and density/variance features are added in later versions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from deup.core.losses import TargetTransform, apply_error_transform, inverse_error_transform
from deup.core.oof import OOFErrorCollector, _safe_index


class DEUPRegressor(RegressorMixin, MetaEstimatorMixin, BaseEstimator):
    """Direct Epistemic Uncertainty Prediction for regression.

    Parameters
    ----------
    base_model:
        The regressor whose uncertainty we estimate. Defaults to
        :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.
    error_model:
        The secondary error predictor ``g``. Defaults to
        :class:`~sklearn.ensemble.HistGradientBoostingRegressor` (no extra deps).
    cv:
        An int (number of ``KFold`` folds) or any splitter exposing ``split``
        (e.g. :class:`deup.splitters.PurgedWalkForward` for time series).
    loss:
        Error-target loss passed to the collector (``"squared"`` by default).
    target_transform:
        Stabilization for ``g``'s regression target: ``"log"`` (default),
        ``"asinh"``, or ``"none"``.
    log_target:
        Deprecated alias for ``target_transform="log"``. If ``False``, sets
        ``target_transform="none"`` unless ``target_transform`` is explicitly given.
    error_eps:
        Stabilizer for ``log`` / ``asinh`` transforms.
    random_state:
        Seed used when ``cv`` is an int (a shuffled ``KFold``).

    Attributes
    ----------
    base_model_ :
        The base model refit on all training data (used for ``predict``).
    error_model_ :
        The fitted error predictor ``g``.
    oof_ :
        The :class:`~deup.core.types.OOFResult` used to train ``g``.
    """

    def __init__(
        self,
        base_model: Any = None,
        error_model: Any = None,
        cv: Any = 5,
        loss: Any = "squared",
        *,
        target_transform: TargetTransform | None = None,
        log_target: bool = True,
        error_eps: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.base_model = base_model
        self.error_model = error_model
        self.cv = cv
        self.loss = loss
        if target_transform is not None:
            self.target_transform: TargetTransform = target_transform
        else:
            self.target_transform = "log" if log_target else "none"
        self.log_target = log_target
        self.error_eps = error_eps
        self.random_state = random_state

    def _resolve_cv(self) -> Any:
        if isinstance(self.cv, int):
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return self.cv

    def fit(self, X: Any, y: npt.ArrayLike, groups: npt.ArrayLike | None = None) -> DEUPRegressor:
        """Fit the base model (out-of-fold) and the error predictor ``g``."""
        base = self.base_model if self.base_model is not None else HistGradientBoostingRegressor()
        err = self.error_model if self.error_model is not None else HistGradientBoostingRegressor()

        collector = OOFErrorCollector(
            base, cv=self._resolve_cv(), loss=self.loss, refit_on_all=True
        )
        oof = collector.fit_collect(X, y, groups=groups)

        assert oof.indices is not None  # collector always records indices
        g_X = _safe_index(X, oof.indices)
        target = apply_error_transform(oof.errors, method=self.target_transform, eps=self.error_eps)

        self.error_model_ = clone(err)
        self.error_model_.fit(g_X, target)
        self.base_model_ = oof.estimator
        self.oof_ = oof
        if hasattr(X, "shape"):
            self.n_features_in_ = int(X.shape[1])
        return self

    def predict(
        self, X: Any, return_uncertainty: bool = False
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Predict, optionally returning ``(prediction, epistemic_uncertainty)``."""
        check_is_fitted(self, "base_model_")
        pred = np.asarray(self.base_model_.predict(X), dtype=float)
        if not return_uncertainty:
            return pred
        return pred, self.predict_epistemic(X)

    def predict_epistemic(self, X: Any) -> npt.NDArray[Any]:
        """Return the estimated epistemic uncertainty ``g(x)`` (>= 0)."""
        check_is_fitted(self, "error_model_")
        raw = np.asarray(self.error_model_.predict(X), dtype=float)
        unc = inverse_error_transform(raw, method=self.target_transform, eps=self.error_eps)
        clipped: npt.NDArray[Any] = np.clip(unc, 0.0, None)
        return clipped
