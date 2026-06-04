"""P-min-core: walk-forward splitter behavior and leakage geometry."""

from __future__ import annotations

import numpy as np
import pytest

from deup.splitters import KFold, PurgedWalkForward, TimeSeriesSplit


def test_reexports_are_usable() -> None:
    assert KFold(n_splits=3).get_n_splits() == 3
    assert TimeSeriesSplit(n_splits=3).get_n_splits() == 3


def test_walk_forward_is_temporal_and_nonoverlapping() -> None:
    X = np.zeros((20, 2))
    splitter = PurgedWalkForward(n_splits=4, embargo=0)
    folds = list(splitter.split(X))
    assert len(folds) == 4
    seen_test: set[int] = set()
    for train_idx, test_idx in folds:
        # train strictly precedes test
        assert train_idx.max() < test_idx.min()
        # test blocks are disjoint across folds
        assert seen_test.isdisjoint(test_idx.tolist())
        seen_test.update(test_idx.tolist())


def test_embargo_creates_a_gap() -> None:
    X = np.zeros((20, 2))
    embargo = 3
    for train_idx, test_idx in PurgedWalkForward(n_splits=3, embargo=embargo).split(X):
        # purged units sit strictly between the last train row and first test row
        assert test_idx.min() - train_idx.max() > embargo


def test_groups_keep_cross_section_together() -> None:
    # 5 dates, 3 assets each -> rows ordered by date
    dates = np.repeat(np.arange(5), 3)
    X = np.zeros((15, 2))
    for train_idx, test_idx in PurgedWalkForward(n_splits=2, embargo=0).split(X, groups=dates):
        train_dates = set(dates[train_idx].tolist())
        test_dates = set(dates[test_idx].tolist())
        # no date is split across train and test
        assert train_dates.isdisjoint(test_dates)
        # every test date appears with its full cross-section (3 rows)
        for d in test_dates:
            assert int((dates[test_idx] == d).sum()) == 3
        # test dates are later than train dates
        assert min(test_dates) > max(train_dates)


def test_too_many_splits_raises() -> None:
    with pytest.raises(ValueError, match="Not enough time units"):
        list(PurgedWalkForward(n_splits=50).split(np.zeros((5, 1))))
