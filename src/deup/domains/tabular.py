"""Generic tabular regression/classification preset.

``TabularDEUP`` wraps :class:`~deup.estimators.DEUPRegressor` or
:class:`~deup.estimators.DEUPClassifier` with ``KFold`` and a Mahalanobis density
feature pipeline — the i.i.d. high-N default path from the architecture map.
"""

from __future__ import annotations

from typing import Any, Literal

from deup.core.features import DensityFeature, FeaturePipeline, RawFeatures
from deup.estimators import DEUPClassifier, DEUPRegressor

TaskKind = Literal["regression", "classification"]


def tabular_feature_pipeline(*, include_raw: bool = True) -> FeaturePipeline:
    """Default g-features for i.i.d. tabular data: raw ``X`` + log-density."""
    parts: list[tuple[str, Any]] = []
    if include_raw:
        parts.append(("raw", RawFeatures()))
    parts.append(("density", DensityFeature(method="mahalanobis")))
    return FeaturePipeline(parts)


class TabularDEUP:
    """Ergonomic tabular preset — delegates to core DEUP estimators.

    Parameters
    ----------
    task:
        ``"regression"`` (default) or ``"classification"``.
    base_model, cv, random_state:
        Forwarded to the underlying estimator. ``cv`` as int uses ``KFold`` /
        ``StratifiedKFold``.
    include_raw:
        If ``True``, concatenate raw ``X`` with density features for ``g``.
    """

    def __init__(
        self,
        base_model: Any = None,
        *,
        task: TaskKind = "regression",
        cv: Any = 5,
        include_raw: bool = True,
        random_state: int | None = None,
    ) -> None:
        features = tabular_feature_pipeline(include_raw=include_raw)
        cls = DEUPClassifier if task == "classification" else DEUPRegressor
        self._estimator = cls(
            base_model=base_model,
            features=features,
            cv=cv,
            random_state=random_state,
        )

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
