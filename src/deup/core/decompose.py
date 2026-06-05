"""Epistemic decomposition, rank-geometry residualization, density ablation.

This module turns the raw error estimate ``g(x)`` into a reported epistemic signal:

- :func:`decompose_epistemic` — ``e_hat(x) = max(0, g(x) - a(x))`` (Lahlou et al.,
  2023, Eq. 9, with the non-negativity clip used in the thesis).
- :class:`RankResidualizer` — removes mechanical rank-geometry coupling from a
  cross-sectional signal (RESEARCH_FINDINGS Finding 3). Required for rankers, where
  ``g`` / the loss target can be partly explained by the within-group rank of the
  model score rather than genuine error.
- :func:`coupling_retention_report` — quantifies coupling reduction and loss-association
  retention before/after residualization.
- :func:`density_kill_criterion` — drops density features that add no signal beyond
  geometry (Finding 3 corollary), with an explicit keep/kill flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression

from deup.core.grouping import Grouping


def decompose_epistemic(
    error: npt.ArrayLike,
    aleatoric: npt.ArrayLike | None = None,
    *,
    clip: bool = True,
) -> npt.NDArray[Any]:
    """Return the epistemic estimate ``e_hat = max(0, g - a)``.

    Parameters
    ----------
    error:
        The error estimate ``g(x)`` (e.g. from :class:`ErrorEstimator`).
    aleatoric:
        The aleatoric estimate ``a(x)``. If ``None``, ``a(x) = 0`` and ``e_hat = g``
        (the paper's conservative proxy, scenario 3).
    clip:
        If ``True`` (default), clip the result at 0 (epistemic uncertainty is
        non-negative).
    """
    g = np.asarray(error, dtype=float)
    if aleatoric is None:
        e_hat = g.copy()
    else:
        a = np.asarray(aleatoric, dtype=float)
        if a.shape != g.shape:
            raise ValueError(f"aleatoric shape {a.shape} != error shape {g.shape}")
        e_hat = g - a
    if clip:
        e_hat = np.clip(e_hat, 0.0, None)
    return np.asarray(e_hat, dtype=float)


class RankResidualizer:
    """Residualize a signal on the within-group rank of a model score.

    For cross-sectional rankers the raw epistemic signal can be partly *mechanical*:
    the within-date rank percentile of ``|score|`` mechanically tracks the loss target
    (Finding 3, per-date rho(e_hat, |score|) ~ 0.616). This transform fits an isotonic
    map from the within-group rank to the signal and subtracts it, leaving the part of
    the signal **not** explained by rank geometry.

    Apply the *same* fitted residualizer to both ``g`` and the loss target to obtain a
    decoupled signal whose association with realized loss can then be measured honestly.

    The axis to rank on is supplied as the ``score`` argument of ``fit``/``transform``
    (pass ``|score|`` to decouple from rank-of-conviction).

    Parameters
    ----------
    out_of_bounds:
        Passed to :class:`~sklearn.isotonic.IsotonicRegression` (default ``"clip"``).
    """

    def __init__(self, *, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = out_of_bounds

    @staticmethod
    def _within_group_rank(
        score: npt.NDArray[Any], groups: npt.ArrayLike | None
    ) -> npt.NDArray[Any]:
        grouping = Grouping.from_labels(groups, score.shape[0])
        return grouping.rank_within(score, pct=True)

    def fit(
        self,
        values: npt.ArrayLike,
        score: npt.ArrayLike,
        groups: npt.ArrayLike | None = None,
    ) -> RankResidualizer:
        """Fit the isotonic rank -> value map (pooled across groups)."""
        v = np.asarray(values, dtype=float)
        s = np.asarray(score, dtype=float)
        ranks = self._within_group_rank(s, groups)
        self.iso_ = IsotonicRegression(out_of_bounds=self.out_of_bounds)
        self.iso_.fit(ranks, v)
        return self

    def transform(
        self,
        values: npt.ArrayLike,
        score: npt.ArrayLike,
        groups: npt.ArrayLike | None = None,
    ) -> npt.NDArray[Any]:
        """Return ``values`` minus the rank-explained component."""
        if not hasattr(self, "iso_"):
            raise RuntimeError("RankResidualizer must be fit before transform")
        v = np.asarray(values, dtype=float)
        s = np.asarray(score, dtype=float)
        ranks = self._within_group_rank(s, groups)
        fitted = np.asarray(self.iso_.predict(ranks), dtype=float)
        return np.asarray(v - fitted, dtype=float)

    def fit_transform(
        self,
        values: npt.ArrayLike,
        score: npt.ArrayLike,
        groups: npt.ArrayLike | None = None,
    ) -> npt.NDArray[Any]:
        return self.fit(values, score, groups).transform(values, score, groups)


@dataclass(frozen=True)
class CouplingReport:
    """Coupling/retention diagnostics for rank residualization."""

    coupling_before: float
    coupling_after: float
    loss_assoc_before: float
    loss_assoc_after: float
    retention: float


def coupling_retention_report(
    g: npt.ArrayLike,
    score: npt.ArrayLike,
    loss: npt.ArrayLike,
    groups: npt.ArrayLike | None = None,
    *,
    residualizer: RankResidualizer | None = None,
) -> CouplingReport:
    """Quantify rank-geometry coupling reduction and loss-association retention.

    Returns Spearman ``rho(g, |score|)`` before/after residualization (coupling) and
    ``rho(signal, loss)`` before/after (loss association). ``retention`` is the ratio
    of after/before loss association (Finding 3 reports R ~ 0.955).
    """
    g_arr = np.asarray(g, dtype=float)
    s_arr = np.asarray(score, dtype=float)
    loss_arr = np.asarray(loss, dtype=float)
    abs_score = np.abs(s_arr)

    # Residualize on the same axis coupling is measured against (|score|), so the
    # mechanical rank-geometry component is the one removed.
    res = residualizer if residualizer is not None else RankResidualizer()
    res.fit(g_arr, abs_score, groups)
    g_resid = res.transform(g_arr, abs_score, groups)
    loss_resid = RankResidualizer().fit_transform(loss_arr, abs_score, groups)

    coupling_before = abs(float(spearmanr(g_arr, abs_score).statistic))
    coupling_after = abs(float(spearmanr(g_resid, abs_score).statistic))
    loss_before = abs(float(spearmanr(g_arr, loss_arr).statistic))
    loss_after = abs(float(spearmanr(g_resid, loss_resid).statistic))
    retention = loss_after / loss_before if loss_before > 0 else 0.0
    return CouplingReport(
        coupling_before=coupling_before,
        coupling_after=coupling_after,
        loss_assoc_before=loss_before,
        loss_assoc_after=loss_after,
        retention=retention,
    )


@dataclass(frozen=True)
class DensityKillDecision:
    """Outcome of the density ablation test."""

    keep: bool
    gain_importance: float
    delta_partial_corr: float
    reason: str


def density_kill_criterion(
    gain_importance: float,
    delta_partial_corr: float,
    *,
    importance_tol: float = 1e-3,
    corr_tol: float = 0.005,
) -> DensityKillDecision:
    """Decide whether to keep density features in ``g``.

    Finding 3 corollary: in homogeneous tabular/finance universes density features can
    add **no signal beyond rank geometry**. Drop density when BOTH its gain importance
    is negligible AND adding it changes the loss partial-correlation by less than
    ``corr_tol``.

    Parameters
    ----------
    gain_importance:
        The density feature's relative gain importance in ``g`` (in ``[0, 1]``).
    delta_partial_corr:
        ``|rho_partial(with density) - rho_partial(without density)|``.
    importance_tol, corr_tol:
        Thresholds below which each signal is considered negligible.

    Returns
    -------
    DensityKillDecision
        ``keep=False`` (kill) iff both signals are negligible.
    """
    negligible_importance = gain_importance < importance_tol
    negligible_corr = abs(delta_partial_corr) < corr_tol
    kill = negligible_importance and negligible_corr
    if kill:
        reason = (
            f"killed: gain_importance {gain_importance:.4g} < {importance_tol} "
            f"and |delta_partial_corr| {abs(delta_partial_corr):.4g} < {corr_tol}"
        )
    elif not negligible_importance and not negligible_corr:
        reason = "kept: both importance and partial-correlation are non-negligible"
    elif not negligible_importance:
        reason = "kept: gain importance is non-negligible"
    else:
        reason = "kept: partial-correlation change is non-negligible"
    return DensityKillDecision(
        keep=not kill,
        gain_importance=gain_importance,
        delta_partial_corr=delta_partial_corr,
        reason=reason,
    )


def partial_correlation(a: npt.ArrayLike, b: npt.ArrayLike, control: npt.ArrayLike) -> float:
    """Pearson partial correlation of ``a`` and ``b`` controlling for ``control``.

    Helper for computing ``delta_partial_corr`` in the density kill criterion: the
    residual correlation of two signals after linearly regressing out a control.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    c_arr = np.asarray(control, dtype=float)
    a_res = a_arr - _linfit_predict(c_arr, a_arr)
    b_res = b_arr - _linfit_predict(c_arr, b_arr)
    # If a residual is negligible relative to its original signal, the control
    # explains essentially all variation -> no partial association.
    a_tol = 1e-9 * (float(np.std(a_arr)) + 1e-12)
    b_tol = 1e-9 * (float(np.std(b_arr)) + 1e-12)
    if float(np.std(a_res)) <= a_tol or float(np.std(b_res)) <= b_tol:
        return 0.0
    return float(pearsonr(a_res, b_res).statistic)


def _linfit_predict(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Least-squares fit of ``y ~ x`` (with intercept); return fitted values."""
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.asarray(design @ coef, dtype=float)
