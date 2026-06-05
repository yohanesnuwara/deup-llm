"""Horizontally stack feature builders for g(x)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _as_2d(block: npt.NDArray[Any]) -> npt.NDArray[Any]:
    if block.ndim == 1:
        return block.reshape(-1, 1)
    return block


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """FeatureUnion-style composition of named transformers.

    Each child is fit on the same ``(X, y)`` and outputs are concatenated column-wise.

    Parameters
    ----------
    transformers:
        List of ``(name, transformer)`` pairs. Each transformer must implement
        ``fit``/``transform`` (sklearn ``TransformerMixin`` convention).
    """

    def __init__(
        self,
        transformers: list[tuple[str, TransformerMixin]] | None = None,
    ) -> None:
        self.transformers = transformers if transformers is not None else []

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> FeaturePipeline:
        if not self.transformers:
            raise ValueError("FeaturePipeline requires at least one transformer.")
        self.transformers_: list[tuple[str, TransformerMixin]] = []
        for name, trans in self.transformers:
            fitted = trans.fit(X, y)
            self.transformers_.append((name, fitted))
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        check_is_fitted(self, "transformers_")
        blocks = [_as_2d(trans.transform(X)) for _, trans in self.transformers_]
        return np.hstack(blocks)

    def get_feature_names_out(
        self, input_features: npt.ArrayLike | None = None
    ) -> npt.NDArray[Any]:
        check_is_fitted(self, "transformers_")
        names: list[str] = []
        for prefix, trans in self.transformers_:
            if not hasattr(trans, "get_feature_names_out"):
                raise AttributeError(f"Transformer '{prefix}' must implement get_feature_names_out")
            sub = trans.get_feature_names_out(input_features)
            names.extend(f"{prefix}__{n}" for n in sub)
        return np.asarray(names, dtype=object)
