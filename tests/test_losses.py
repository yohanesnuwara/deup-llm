"""P-min-core: error-target losses."""

from __future__ import annotations

import numpy as np
import pytest

from deup.core import get_loss


def test_squared_error() -> None:
    fn = get_loss("squared")
    err = fn(np.array([1.0, 2.0]), np.array([1.5, 0.0]))
    assert np.allclose(err, [0.25, 4.0])


def test_absolute_error() -> None:
    fn = get_loss("absolute")
    err = fn(np.array([1.0, 2.0]), np.array([1.5, 0.0]))
    assert np.allclose(err, [0.5, 2.0])


def test_logloss_binary_is_nonnegative() -> None:
    fn = get_loss("logloss")
    err = fn(np.array([1, 0, 1]), np.array([0.9, 0.2, 0.4]))
    assert np.all(err >= 0)
    # a confident correct prediction has lower loss than a hesitant one
    assert err[0] < err[2]


def test_logloss_multiclass() -> None:
    fn = get_loss("logloss")
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    err = fn(np.array([0, 2]), probs)
    assert np.allclose(err, -np.log([0.7, 0.8]))


def test_rank_loss_perfect_ranking_is_zero() -> None:
    fn = get_loss("rank")
    groups = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([1.0, 2.0, 3.0, 9.0, 8.0, 7.0])
    pred = y.copy()  # identical ranking within each group
    err = fn(y, pred, groups)
    assert np.allclose(err, 0.0)


def test_rank_loss_in_unit_interval() -> None:
    fn = get_loss("rank")
    groups = np.array([0, 0, 0, 0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([4.0, 3.0, 2.0, 1.0])  # reversed
    err = fn(y, pred, groups)
    assert np.all((err >= 0) & (err <= 1))
    assert err.sum() > 0


def test_callable_escape_hatch() -> None:
    fn = get_loss(lambda yt, yp, groups=None: np.abs(np.asarray(yt) - np.asarray(yp)))
    assert np.allclose(fn(np.array([3.0]), np.array([1.0])), [2.0])


def test_unknown_loss_raises() -> None:
    with pytest.raises(ValueError, match="Unknown loss"):
        get_loss("not-a-loss")
