"""Cross-validation splitters for collecting out-of-sample errors.

DEUP's error predictor must be trained on *out-of-sample* errors of the base model
(Lahlou et al., 2023, Algorithms 1-2). The splitter is therefore the leakage-control
knob: choose ``KFold`` for i.i.d. data, ``TimeSeriesSplit`` for ordered data, and
:class:`PurgedWalkForward` for time-series / cross-sectional data where an embargo is
needed to prevent look-ahead between train and test.

``KFold`` and ``TimeSeriesSplit`` are re-exported from scikit-learn so users have a
single import surface.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import KFold, TimeSeriesSplit

__all__ = ["KFold", "PurgedWalkForward", "TimeSeriesSplit"]


def _n_rows(X: Any) -> int:
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return len(X)


class PurgedWalkForward:
    """Expanding-window walk-forward splitter with an embargo (purge).

    Time is measured in *units*. If ``groups`` is passed to :meth:`split`, each
    unique group value (e.g. a date) is one time unit and the whole cross-section
    of a unit always stays together in the same fold — which is required for
    cross-sectional rank losses. If ``groups`` is ``None``, each row is its own unit.

    For each of ``n_splits`` folds, the test block is a contiguous range of the most
    recent units; the training set is all units strictly before it, minus an
    ``embargo`` of units immediately preceding the test block (the purge). This
    prevents the base model from being trained on data adjacent to (and potentially
    leaking into) the evaluation block.

    Parameters
    ----------
    n_splits:
        Number of walk-forward test folds.
    embargo:
        Number of time units to drop between the train set and each test block.
    min_train_size:
        Minimum number of training units required to emit a fold; smaller folds are
        skipped.
    max_train_size:
        If set, training uses at most this many of the most recent units (rolling
        window). Otherwise the window expands from the start.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo: int = 0,
        min_train_size: int = 1,
        max_train_size: int | None = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        if min_train_size < 1:
            raise ValueError("min_train_size must be >= 1")
        self.n_splits = n_splits
        self.embargo = embargo
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: npt.ArrayLike | None = None
    ) -> Iterator[tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
        """Yield ``(train_idx, test_idx)`` row-index arrays for each fold."""
        n = _n_rows(X)
        if groups is None:
            row_units = np.arange(n)
            n_units = n
        else:
            groups_arr = np.asarray(groups)
            if groups_arr.shape[0] != n:
                raise ValueError(f"groups length {groups_arr.shape[0]} != n_rows {n}")
            _, row_units = np.unique(groups_arr, return_inverse=True)
            row_units = np.ravel(row_units)
            n_units = int(row_units.max()) + 1 if n > 0 else 0

        test_size = n_units // (self.n_splits + 1)
        if test_size < 1:
            raise ValueError(f"Not enough time units ({n_units}) for n_splits={self.n_splits}")

        indices = np.arange(n)
        for i in range(self.n_splits):
            test_start = n_units - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            train_end = test_start - self.embargo
            if train_end < self.min_train_size:
                continue
            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)

            train_mask = (row_units >= train_start) & (row_units < train_end)
            test_mask = (row_units >= test_start) & (row_units < test_end)
            train_idx = indices[train_mask]
            test_idx = indices[test_mask]
            if train_idx.shape[0] == 0 or test_idx.shape[0] == 0:
                continue
            yield train_idx, test_idx
