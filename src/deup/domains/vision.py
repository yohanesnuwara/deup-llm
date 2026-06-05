"""Vision / OOD classification preset.

``VisionDEUP`` wraps :class:`~deup.estimators.DEUPClassifier` with an
embedding → density + variance feature path for ``g(x)``, matching the CIFAR-10-C
architecture map (high-N i.i.d. batches where individual-level ``g`` saturates).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from deup.core.features import DensityFeature, VarianceFeature
from deup.core.features.density import DensityMethod
from deup.estimators import DEUPClassifier


class IdentityEmbedding(BaseEstimator, TransformerMixin):
    """Flatten tensors to 2-D embeddings (test / baseline without a CNN)."""

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> IdentityEmbedding:
        arr = np.asarray(X)
        sample = self._embed(arr[:1])
        self.n_features_out_ = int(sample.shape[1])
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "n_features_out_")
        return self._embed(X)

    @staticmethod
    def _embed(X: npt.ArrayLike) -> npt.NDArray[Any]:
        arr = np.asarray(X, dtype=float)
        if arr.ndim <= 2:
            return arr
        return arr.reshape(arr.shape[0], -1)


class EmbeddingUncertaintyFeatures(BaseEstimator, TransformerMixin):
    """Embed inputs, then append density + variance features (vision preset glue)."""

    def __init__(
        self,
        embedding: BaseEstimator | Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
        *,
        density_method: DensityMethod = "mahalanobis",
        variance_estimators: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.embedding = embedding
        self.density_method = density_method
        self.variance_estimators = variance_estimators
        self.random_state = random_state

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> EmbeddingUncertaintyFeatures:
        self.embedder_ = self._resolve_embedder()
        if hasattr(self.embedder_, "fit"):
            self.embedder_.fit(X, y)
        z = self._transform_embed(X)
        self.density_ = DensityFeature(method=self.density_method).fit(z)
        self.variance_ = VarianceFeature(
            method="ensemble",
            n_estimators=self.variance_estimators,
            random_state=self.random_state,
        ).fit(z, y)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "density_")
        z = self._transform_embed(X)
        dens = self.density_.transform(z)
        var = self.variance_.transform(z)
        return np.hstack([dens, var])

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[Any]:
        check_is_fitted(self, "density_")
        return np.array(["log_density", "log_variance"], dtype=object)

    def _resolve_embedder(self) -> BaseEstimator | IdentityEmbedding:
        if self.embedding is None:
            return IdentityEmbedding()
        if callable(self.embedding):
            return _CallableEmbedding(self.embedding)
        return self.embedding

    def _transform_embed(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        out = self.embedder_.transform(X)
        return np.asarray(out, dtype=float)


class _CallableEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, fn: Callable[[npt.ArrayLike], npt.ArrayLike]) -> None:
        self.fn = fn

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> _CallableEmbedding:
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        return np.asarray(self.fn(X), dtype=float)


class VisionDEUP:
    """Classification preset: embedding → density + variance → ``g(x)``.

    Parameters
    ----------
    embedding:
        Optional sklearn transformer or callable mapping raw inputs to embeddings.
        Defaults to :class:`IdentityEmbedding` (flatten tensors). Inputs are embedded
        once at the API boundary so the base classifier always sees 2-D arrays.
    cv:
        ``KFold`` folds when int (i.i.d. vision batches).
    """

    def __init__(
        self,
        base_model: Any = None,
        *,
        embedding: BaseEstimator | Callable[[npt.ArrayLike], npt.ArrayLike] | None = None,
        cv: Any = 5,
        random_state: int | None = None,
    ) -> None:
        self._embedding = embedding
        self._estimator = DEUPClassifier(
            base_model=base_model,
            features=EmbeddingUncertaintyFeatures(embedding=embedding, random_state=random_state),
            cv=cv,
            random_state=random_state,
        )

    def _prepare_X(self, X: Any, *, fit: bool = False) -> npt.NDArray[Any]:
        if fit or not hasattr(self, "_input_embedder_"):
            if self._embedding is None:
                embedder: BaseEstimator | _CallableEmbedding = IdentityEmbedding()
            elif callable(self._embedding) and not hasattr(self._embedding, "transform"):
                embedder = _CallableEmbedding(self._embedding)
            else:
                embedder = self._embedding
            if fit:
                embedder.fit(X)
            self._input_embedder_ = embedder
        out = self._input_embedder_.transform(X)
        return np.asarray(out, dtype=float)

    def fit(self, X: Any, y: Any, **kwargs: Any) -> VisionDEUP:
        self._estimator.fit(self._prepare_X(X, fit=True), y, **kwargs)
        return self

    def predict(self, X: Any, **kwargs: Any) -> Any:
        return self._estimator.predict(self._prepare_X(X), **kwargs)

    def predict_proba(self, X: Any) -> Any:
        return self._estimator.predict_proba(self._prepare_X(X))

    def predict_epistemic(self, X: Any) -> Any:
        return self._estimator.predict_epistemic(self._prepare_X(X))

    @property
    def estimator(self) -> DEUPClassifier:
        return self._estimator
