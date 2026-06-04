"""Error-target losses for DEUP.

In DEUP the *error target* for a row is the base model's pointwise loss
``l(y, f(x))`` (Lahlou et al., 2023, Eq. 9 / Alg. 1). The secondary predictor ``g``
then regresses these targets. This module provides the common choices behind a
single ``get_loss`` factory, plus a ``callable`` escape hatch for custom losses.

Each loss has the signature ``loss(y_true, y_pred, groups=None) -> ndarray`` and
returns one non-negative error per row. ``groups`` is only used by group-aware losses
such as :func:`rank_loss`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from deup.core.grouping import Grouping

LossFn = Callable[..., npt.NDArray[Any]]


def squared_error(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: npt.ArrayLike | None = None
) -> npt.NDArray[Any]:
    """Squared residual ``(y - f(x))**2`` (regression)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    out: npt.NDArray[Any] = (yt - yp) ** 2
    return out


def absolute_error(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: npt.ArrayLike | None = None
) -> npt.NDArray[Any]:
    """Absolute residual ``|y - f(x)|`` (regression)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    out: npt.NDArray[Any] = np.abs(yt - yp)
    return out


def log_loss(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: npt.ArrayLike | None = None
) -> npt.NDArray[Any]:
    """Pointwise cross-entropy (classification).

    ``y_pred`` may be a 1-D vector of positive-class probabilities (binary) or a 2-D
    array of class probabilities (multiclass); ``y_true`` holds class indices
    (or 0/1 for binary).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred, dtype=float)
    eps = 1e-12
    if yp.ndim == 2:
        probs = np.clip(yp, eps, 1.0)
        true_idx = yt.astype(int)
        chosen = probs[np.arange(probs.shape[0]), true_idx]
        multiclass: npt.NDArray[Any] = -np.log(chosen)
        return multiclass
    p = np.clip(yp, eps, 1.0 - eps)
    yt_f = yt.astype(float)
    binary: npt.NDArray[Any] = -(yt_f * np.log(p) + (1.0 - yt_f) * np.log(1.0 - p))
    return binary


def rank_loss(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: npt.ArrayLike | None = None
) -> npt.NDArray[Any]:
    """Per-group absolute rank displacement (cross-sectional ranking).

    For each group (e.g. a date), ranks ``y_true`` and ``y_pred`` to percentiles and
    returns ``|rank_pct(y_true) - rank_pct(y_pred)|``. This is the error target used
    by the thesis's cross-sectional ranker. Requires a group-coherent splitter so
    that each group's full cross-section is scored together.
    """
    yt = np.asarray(y_true, dtype=float)
    grouping = Grouping.from_labels(groups, n=yt.shape[0])
    rank_true = grouping.rank_within(yt, pct=True)
    rank_pred = grouping.rank_within(y_pred, pct=True)
    out: npt.NDArray[Any] = np.abs(rank_true - rank_pred)
    return out


_REGISTRY: dict[str, LossFn] = {
    "squared": squared_error,
    "absolute": absolute_error,
    "logloss": log_loss,
    "rank": rank_loss,
}


def get_loss(loss: str | LossFn) -> LossFn:
    """Resolve ``loss`` (a registry name or a callable) to a loss function."""
    if callable(loss):
        return loss
    try:
        return _REGISTRY[loss]
    except KeyError:
        raise ValueError(
            f"Unknown loss {loss!r}. Choose from {sorted(_REGISTRY)} or pass a callable."
        ) from None
