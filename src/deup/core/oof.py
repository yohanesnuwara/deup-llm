"""Out-of-fold error collection -- the correctness heart of DEUP.

This implements the paper's Algorithm 2 (K-fold pre-fill of the error dataset): for
each fold, fit a *fresh* clone of the base model on the training rows and predict the
held-out rows, so every row receives an **out-of-sample** prediction. The pointwise
loss of those predictions is the training target for the secondary error predictor.

Training the error predictor on in-sample errors instead (e.g. by predicting rows the
base model was trained on) is the canonical DEUP failure mode -- it underestimates
epistemic uncertainty (Lahlou et al., 2023, Sec. 3.2). The leakage test in the test
suite is designed to fail if this collector ever regresses to in-sample behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from deup.core.losses import LossFn, get_loss
from deup.core.types import OOFResult


def _safe_index(X: Any, idx: npt.NDArray[Any]) -> Any:
    """Row-index ``X`` whether it is a numpy array or a pandas object."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return np.asarray(X)[idx]


class OOFErrorCollector:
    """Collect a base model's out-of-fold predictions and pointwise errors.

    Parameters
    ----------
    estimator:
        The base model ``f`` (any scikit-learn-style ``fit``/``predict`` object).
        It is cloned per fold; the passed instance is never fitted in place.
    cv:
        A splitter exposing ``split(X, y, groups)`` (e.g. ``KFold``,
        ``TimeSeriesSplit``, or :class:`deup.splitters.PurgedWalkForward`).
    loss:
        Error-target loss: a registry name (``"squared"``, ``"absolute"``,
        ``"logloss"``, ``"rank"``) or a callable ``loss(y_true, y_pred, groups)``.
    proba:
        If ``True``, use ``predict_proba`` (positive-class column for binary) instead
        of ``predict`` -- required for classification log-loss targets.
    refit_on_all:
        If ``True`` (default), also refit a clone of the base model on all data and
        expose it as ``OOFResult.estimator`` for deployment.

    Notes
    -----
    Rows never assigned to a test fold (e.g. the earliest rows under walk-forward)
    are excluded from the returned :class:`~deup.core.types.OOFResult`.
    """

    def __init__(
        self,
        estimator: Any,
        cv: Any,
        loss: str | LossFn = "squared",
        *,
        proba: bool = False,
        refit_on_all: bool = True,
    ) -> None:
        self.estimator = estimator
        self.cv = cv
        self.loss = loss
        self.proba = proba
        self.refit_on_all = refit_on_all

    def fit_collect(
        self, X: Any, y: npt.ArrayLike, groups: npt.ArrayLike | None = None
    ) -> OOFResult:
        """Run the out-of-fold loop and return the collected errors."""
        y_arr = np.asarray(y)
        n = y_arr.shape[0]
        groups_arr = None if groups is None else np.asarray(groups)

        oof_pred = np.full(n, np.nan, dtype=float)
        fold_ids = np.full(n, -1, dtype=np.intp)

        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X, y_arr, groups_arr)):
            model = clone(self.estimator)
            model.fit(_safe_index(X, train_idx), y_arr[train_idx])
            X_test = _safe_index(X, test_idx)
            if self.proba:
                proba = np.asarray(model.predict_proba(X_test), dtype=float)
                pred = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
            else:
                pred = np.asarray(model.predict(X_test), dtype=float)
            oof_pred[test_idx] = pred
            fold_ids[test_idx] = fold

        mask = fold_ids >= 0
        if not mask.any():
            raise ValueError("No out-of-fold predictions were produced by the splitter.")

        loss_fn = get_loss(self.loss)
        g_groups = None if groups_arr is None else groups_arr[mask]
        errors = np.asarray(loss_fn(y_arr[mask], oof_pred[mask], g_groups), dtype=float)

        estimator_ = None
        if self.refit_on_all:
            estimator_ = clone(self.estimator)
            estimator_.fit(X, y_arr)

        return OOFResult(
            predictions=oof_pred[mask],
            errors=errors,
            fold_ids=fold_ids[mask],
            group_ids=g_groups,
            estimator=estimator_,
        )
