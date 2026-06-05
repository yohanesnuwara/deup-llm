"""User-facing DEUP estimators.

Three task-specific wrappers compose the core pipeline:

    OOFErrorCollector  ->  ErrorEstimator (g)  ->  optional aleatoric (a)
                                                    ->  decompose_epistemic

Each exposes the ergonomic API::

    model.fit(X, y)
    pred, unc = model.predict(X, return_uncertainty=True)
    top_k = model.acquire(pool, k=5)

See :doc:`getting-started` and :doc:`decomposition` for usage guides.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    clone,
)
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from deup.calibration.conformal import (
    ConformalMethod,
    ConformalResult,
    UncertaintyCalibrator,
)
from deup.core.decompose import RankResidualizer, decompose_epistemic
from deup.core.error_estimator import ErrorEstimator
from deup.core.losses import TargetTransform
from deup.core.oof import OOFErrorCollector, _safe_index
from deup.splitters import PurgedWalkForward


class _DEUPBase(MetaEstimatorMixin, BaseEstimator):
    """Shared DEUP fit/predict logic for regression, classification, and ranking."""

    _task: str = "regression"
    _default_loss: str = "squared"
    _proba: bool = False

    def __init__(
        self,
        base_model: Any = None,
        error_model: Any = None,
        features: Any = None,
        aleatoric: Any = None,
        cv: Any = 5,
        loss: Any | None = None,
        *,
        target_transform: TargetTransform | None = None,
        log_target: bool = True,
        error_eps: float = 1e-6,
        decompose: bool = False,
        residualize_rank: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.base_model = base_model
        self.error_model = error_model
        self.features = features
        self.aleatoric = aleatoric
        self.cv = cv
        self.loss = loss
        if target_transform is not None:
            self.target_transform: TargetTransform = target_transform
        else:
            self.target_transform = "log" if log_target else "none"
        self.log_target = log_target
        self.error_eps = error_eps
        self.decompose = decompose
        self.residualize_rank = residualize_rank
        self.random_state = random_state

    def _resolve_loss(self) -> Any:
        return self.loss if self.loss is not None else self._default_loss

    def _resolve_cv(self, y: npt.ArrayLike | None = None) -> Any:
        if not isinstance(self.cv, int):
            return self.cv
        if self._task == "classification" and y is not None:
            return StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

    def _resolve_base_model(self) -> Any:
        if self.base_model is not None:
            return self.base_model
        if self._task == "classification":
            return HistGradientBoostingClassifier()
        return HistGradientBoostingRegressor()

    def _build_error_estimator(self) -> ErrorEstimator:
        return ErrorEstimator(
            model=self.error_model,
            features=self.features,
            target_transform=self.target_transform,
            error_eps=self.error_eps,
        )

    def fit(
        self,
        X: Any,
        y: npt.ArrayLike,
        groups: npt.ArrayLike | None = None,
    ) -> _DEUPBase:
        if self._task == "ranking" and groups is None:
            raise ValueError("DEUPRanker requires `groups` (e.g. cross-section dates) at fit.")

        base = self._resolve_base_model()
        collector = OOFErrorCollector(
            base,
            cv=self._resolve_cv(y),
            loss=self._resolve_loss(),
            proba=self._proba,
            refit_on_all=True,
        )
        oof = collector.fit_collect(X, y, groups=groups)
        assert oof.indices is not None

        g_X = _safe_index(X, oof.indices)
        g_y = np.asarray(y)[oof.indices] if y is not None else None

        self.error_estimator_ = self._build_error_estimator()
        self.error_estimator_.fit(g_X, oof.errors, y=g_y)

        self.aleatoric_ = None
        if self.aleatoric is not None:
            self.aleatoric_ = clone(self.aleatoric)
            self.aleatoric_.fit(X, y)

        self.base_model_ = oof.estimator
        self.oof_ = oof
        self.groups_ = None if groups is None else np.asarray(groups)
        self._residualizer_: RankResidualizer | None = None

        if self.residualize_rank:
            scores = np.asarray(self.base_model_.predict(g_X), dtype=float)
            g_raw = self.error_estimator_.predict(g_X)
            fit_groups = None
            if groups is not None:
                fit_groups = np.asarray(groups)[oof.indices]
            self._residualizer_ = RankResidualizer().fit(g_raw, np.abs(scores), groups=fit_groups)

        if hasattr(X, "shape"):
            self.n_features_in_ = int(X.shape[1])
        return self

    def _predict_g(self, X: Any, groups: npt.ArrayLike | None = None) -> npt.NDArray[Any]:
        check_is_fitted(self, "error_estimator_")
        g = self.error_estimator_.predict(X)
        if self._residualizer_ is not None:
            scores = np.asarray(self.base_model_.predict(X), dtype=float)
            g = self._residualizer_.transform(g, np.abs(scores), groups=groups)
            g = np.clip(g, 0.0, None)
        return g

    def _predict_a(self, X: Any) -> npt.NDArray[Any] | None:
        if self.aleatoric_ is None:
            return None
        return np.asarray(self.aleatoric_.predict(X), dtype=float)

    def predict_epistemic(self, X: Any, groups: npt.ArrayLike | None = None) -> npt.NDArray[Any]:
        """Return the estimated epistemic uncertainty ``e_hat(x)`` (>= 0).

        Parameters
        ----------
        groups:
            Per-row group labels (required for correct rank residualization in
            :class:`DEUPRanker` on panel data).
        """
        g = self._predict_g(X, groups=groups)
        if self.decompose or self.aleatoric_ is not None:
            return decompose_epistemic(g, self._predict_a(X))
        return g

    def acquire(
        self,
        X: Any,
        k: int = 1,
        *,
        groups: npt.ArrayLike | None = None,
        return_uncertainty: bool = False,
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Return indices of the ``k`` pool points with highest epistemic uncertainty.

        Active-learning hook from the DEUP paper (Sec. 3.2): select the inputs where
        the model expects to learn the most.

        Parameters
        ----------
        X:
            Candidate pool (same feature space as training data).
        k:
            Number of points to acquire.
        return_uncertainty:
            If ``True``, also return the epistemic values at the selected points.
        """
        unc = self.predict_epistemic(X, groups=groups)
        k_eff = min(k, len(unc))
        idx = np.argpartition(-unc, k_eff - 1)[:k_eff]
        # Sort selected indices by uncertainty descending for deterministic ordering.
        idx = idx[np.argsort(-unc[idx])]
        if return_uncertainty:
            return idx, unc[idx]
        return idx

    def calibrate(
        self,
        X: Any,
        y: npt.ArrayLike,
        *,
        method: ConformalMethod = "normalized",
        alpha: float = 0.1,
        groups: npt.ArrayLike | None = None,
    ) -> _DEUPBase:
        """Fit a split-conformal calibrator on a **held-out** set.

        Call after :meth:`fit`, on data the base model and ``g`` did not see, to enable
        :meth:`predict_interval` with finite-sample marginal coverage ``1 - alpha``.

        Parameters
        ----------
        X, y:
            Held-out calibration inputs and targets.
        method:
            ``"normalized"`` (DEUP-scaled, default) or ``"mondrian"`` (per-group).
        alpha:
            Miscoverage level; target coverage is ``1 - alpha``.
        groups:
            Per-row group labels (required for ``method="mondrian"``; also used for
            ranker residualization).
        """
        check_is_fitted(self, "base_model_")
        y_pred = np.asarray(self.base_model_.predict(X), dtype=float)
        unc = self.predict_epistemic(X, groups=groups)
        self.calibrator_ = UncertaintyCalibrator(method=method, alpha=alpha)
        self.calibrator_.fit(y, y_pred, unc, groups=groups)
        return self

    def predict_interval(
        self,
        X: Any,
        *,
        groups: npt.ArrayLike | None = None,
    ) -> ConformalResult:
        """Return calibrated prediction intervals (requires :meth:`calibrate` first)."""
        check_is_fitted(self, "calibrator_")
        y_pred = np.asarray(self.base_model_.predict(X), dtype=float)
        unc = self.predict_epistemic(X, groups=groups)
        return self.calibrator_.predict_interval(y_pred, unc, groups=groups)


class DEUPRegressor(_DEUPBase, RegressorMixin):
    """Direct Epistemic Uncertainty Prediction for regression.

    Parameters
    ----------
    base_model:
        The regressor whose uncertainty we estimate. Defaults to
        :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.
    error_model:
        Secondary error predictor ``g``. Defaults to HGB regressor.
    features:
        Optional :class:`~deup.core.features.FeaturePipeline` for stationarizing
        features (default: raw ``X``).
    aleatoric:
        Optional aleatoric estimator ``a(x)`` (e.g.
        :class:`~deup.core.aleatoric.Heteroscedastic`). When set, ``predict_epistemic``
        returns ``max(0, g - a)``.
    cv:
        An int (``KFold`` folds) or any splitter with ``split(X, y, groups)``.
    loss:
        Error-target loss (``"squared"`` by default).
    target_transform:
        Stabilization for ``g``'s target: ``"log"`` (default), ``"asinh"``, ``"none"``.
    decompose:
        If ``True``, always apply ``max(0, g - a)`` even when ``aleatoric`` is ``None``
        (with ``a=0`` this is a no-op).
    random_state:
        Seed when ``cv`` is an int.

    Attributes
    ----------
    base_model_, error_estimator_, oof_, aleatoric_ :
        Fitted components.
    """

    _task = "regression"
    _default_loss = "squared"
    _proba = False

    def predict(
        self,
        X: Any,
        return_uncertainty: bool = False,
        groups: npt.ArrayLike | None = None,
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Predict, optionally returning ``(prediction, epistemic_uncertainty)``."""
        check_is_fitted(self, "base_model_")
        pred = np.asarray(self.base_model_.predict(X), dtype=float)
        if not return_uncertainty:
            return pred
        return pred, self.predict_epistemic(X, groups=groups)


class DEUPClassifier(_DEUPBase, ClassifierMixin):
    """Direct Epistemic Uncertainty Prediction for classification.

    Uses ``predict_proba`` for OOF error collection and defaults to ``logloss`` as
    the error target.

    Parameters
    ----------
    base_model:
        Classifier whose uncertainty we estimate. Defaults to HGB classifier.
    loss:
        ``"logloss"`` (default) or ``"brier"``.
    cv:
        An int (``StratifiedKFold``) or custom splitter.
    features, aleatoric, target_transform, error_eps, decompose, random_state:
        Same as :class:`DEUPRegressor`.
    """

    _task = "classification"
    _default_loss = "logloss"
    _proba = True

    def predict(
        self,
        X: Any,
        return_uncertainty: bool = False,
        groups: npt.ArrayLike | None = None,
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        check_is_fitted(self, "base_model_")
        pred = np.asarray(self.base_model_.predict(X))
        if not return_uncertainty:
            return pred
        return pred, self.predict_epistemic(X, groups=groups)

    def predict_proba(self, X: Any) -> npt.NDArray[Any]:
        check_is_fitted(self, "base_model_")
        return np.asarray(self.base_model_.predict_proba(X), dtype=float)


class DEUPRanker(_DEUPBase, RegressorMixin):
    """Direct Epistemic Uncertainty Prediction for cross-sectional ranking.

    Defaults to ``loss="rank"`` and ``residualize_rank=True`` so the reported signal
    is decoupled from mechanical rank geometry (Finding 3). Requires ``groups`` at
    ``fit`` (e.g. dates). Defaults to :class:`~deup.splitters.PurgedWalkForward` when
    ``cv`` is an int.

    Parameters
    ----------
    base_model:
        The ranker / regressor whose ordering uncertainty we estimate.
    residualize_rank:
        If ``True`` (default), apply isotonic rank-geometry residualization to ``g``.
    features, aleatoric, target_transform, error_eps, decompose, random_state:
        Same as :class:`DEUPRegressor`.

    Notes
    -----
    ``groups`` is required at ``fit(X, y, groups=...)`` and should be passed to
    ``predict(..., groups=...)`` / ``predict_epistemic(..., groups=...)`` at inference
    for correct within-date residualization.
    """

    _task = "ranking"
    _default_loss = "rank"
    _proba = False

    def __init__(
        self,
        base_model: Any = None,
        error_model: Any = None,
        features: Any = None,
        aleatoric: Any = None,
        cv: Any = 5,
        loss: Any | None = None,
        *,
        target_transform: TargetTransform | None = None,
        log_target: bool = True,
        error_eps: float = 1e-6,
        decompose: bool = False,
        residualize_rank: bool = True,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            base_model=base_model,
            error_model=error_model,
            features=features,
            aleatoric=aleatoric,
            cv=cv,
            loss=loss,
            target_transform=target_transform,
            log_target=log_target,
            error_eps=error_eps,
            decompose=decompose,
            residualize_rank=residualize_rank,
            random_state=random_state,
        )

    def _resolve_cv(self, y: npt.ArrayLike | None = None) -> Any:
        if isinstance(self.cv, int):
            return PurgedWalkForward(n_splits=self.cv, embargo=0)
        return self.cv

    def predict(
        self,
        X: Any,
        return_uncertainty: bool = False,
        groups: npt.ArrayLike | None = None,
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        check_is_fitted(self, "base_model_")
        pred = np.asarray(self.base_model_.predict(X), dtype=float)
        if not return_uncertainty:
            return pred
        return pred, self.predict_epistemic(X, groups=groups)
