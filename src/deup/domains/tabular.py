"""Generic tabular regression/classification preset.

``TabularDEUP`` wraps :class:`~deup.estimators.DEUPRegressor` or
:class:`~deup.estimators.DEUPClassifier` with ``KFold`` and a Mahalanobis density
feature pipeline — the i.i.d. high-N default path from the architecture map.

Optional ``backend`` wires gradient-boosting base + error predictors (LightGBM,
XGBoost, CatBoost) without changing the five-axis defaults.
"""

from __future__ import annotations

from typing import Any, Literal

from deup.core.features import DensityFeature, FeaturePipeline, RawFeatures
from deup.estimators import DEUPClassifier, DEUPRegressor

TaskKind = Literal["regression", "classification"]
BackendKind = Literal["sklearn", "lgbm", "xgb", "catboost"]


def tabular_feature_pipeline(*, include_raw: bool = True) -> FeaturePipeline:
    """Default g-features for i.i.d. tabular data: raw ``X`` + log-density."""
    parts: list[tuple[str, Any]] = []
    if include_raw:
        parts.append(("raw", RawFeatures()))
    parts.append(("density", DensityFeature(method="mahalanobis")))
    return FeaturePipeline(parts)


def default_tabular_base_model(
    backend: BackendKind = "sklearn",
    *,
    task: TaskKind = "regression",
    random_state: int | None = None,
) -> Any:
    """Default ``f`` model for a tabular backend."""
    if backend == "sklearn":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        if task == "classification":
            return HistGradientBoostingClassifier(random_state=random_state)
        return HistGradientBoostingRegressor(random_state=random_state)

    if backend == "lgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise ImportError('backend="lgbm" requires: pip install "deup[gbm]"') from exc
        cls = LGBMClassifier if task == "classification" else LGBMRegressor
        return cls(n_estimators=100, random_state=random_state, verbose=-1)

    if backend == "xgb":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError('backend="xgb" requires: pip install "deup[xgb]"') from exc
        cls = XGBClassifier if task == "classification" else XGBRegressor
        kwargs: dict[str, Any] = {
            "n_estimators": 100,
            "random_state": random_state,
            "verbosity": 0,
        }
        if task == "classification":
            kwargs["eval_metric"] = "logloss"
        return cls(**kwargs)

    if backend == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:
            raise ImportError('backend="catboost" requires: pip install "deup[catboost]"') from exc
        cls = CatBoostClassifier if task == "classification" else CatBoostRegressor
        return cls(
            iterations=100,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )

    raise ValueError(f"Unknown backend: {backend!r}")


def default_tabular_error_model(backend: BackendKind) -> str | None:
    """Error-predictor ``g`` spec for ``ErrorEstimator`` (``None`` = sklearn default)."""
    if backend == "sklearn":
        return None
    return backend


class TabularDEUP:
    """Ergonomic tabular preset — delegates to core DEUP estimators.

    Parameters
    ----------
    base_model:
        Primary predictor ``f``. When ``None``, chosen from ``backend``.
    backend:
        ``"sklearn"`` (default), ``"lgbm"``, ``"xgb"``, or ``"catboost"``. Sets default
        base + error models when those are ``None``.
    error_model:
        Secondary error predictor ``g``. When ``None`` and ``backend`` is a GBM, uses the
        same family (e.g. LightGBM for ``backend="lgbm"``).
    task:
        ``"regression"`` (default) or ``"classification"``.
    cv, random_state, include_raw:
        Forwarded / used as in the core estimator.
    """

    def __init__(
        self,
        base_model: Any = None,
        *,
        backend: BackendKind = "sklearn",
        error_model: Any = None,
        task: TaskKind = "regression",
        cv: Any = 5,
        include_raw: bool = True,
        random_state: int | None = None,
    ) -> None:
        if base_model is None:
            base_model = default_tabular_base_model(
                backend, task=task, random_state=random_state
            )
        if error_model is None:
            error_model = default_tabular_error_model(backend)

        features = tabular_feature_pipeline(include_raw=include_raw)
        cls = DEUPClassifier if task == "classification" else DEUPRegressor
        self._backend = backend
        self._estimator = cls(
            base_model=base_model,
            error_model=error_model,
            features=features,
            cv=cv,
            random_state=random_state,
        )

    @property
    def backend(self) -> BackendKind:
        """Configured gradient-boosting / sklearn backend."""
        return self._backend

    def fit(self, X: Any, y: Any, **kwargs: Any) -> TabularDEUP:
        self._estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any, **kwargs: Any) -> Any:
        return self._estimator.predict(X, **kwargs)

    def predict_epistemic(self, X: Any) -> Any:
        return self._estimator.predict_epistemic(X)

    def calibrate(self, X: Any, y: Any, **kwargs: Any) -> TabularDEUP:
        self._estimator.calibrate(X, y, **kwargs)
        return self

    def predict_interval(self, X: Any, **kwargs: Any) -> Any:
        return self._estimator.predict_interval(X, **kwargs)

    @property
    def estimator(self) -> DEUPRegressor | DEUPClassifier:
        """Underlying core estimator (for advanced composition)."""
        return self._estimator
