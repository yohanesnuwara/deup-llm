"""Aggregation-reliability diagnostic (Finding 1: the N / i.i.d. law).

Aggregating a per-item epistemic signal into a context-level signal
``agg_g(c) = mean_{i in c} g(x_i)`` is a *consistent* estimator of the mean context
error -- but only as ``N -> large`` and only if the within-context errors are
exchangeable. With small N and temporal-regime dependence, the estimator's variance
and bias swamp the signal.

Empirical reference points (Sanderink, 2026), to be read as orientation, **not** as a
hard promise for a user's data:

    AUROC(agg_g, bad context) ~ 0.55  at N ~ 50  (non-i.i.d. finance cross-sections)
    AUROC(agg_g, bad context) ~ 0.955 at N ~ 10,000 (i.i.d. vision batches)

This module estimates, per context, an **effective sample size** that discounts N by
within-context autocorrelation, and emits a warning when aggregation is unlikely to be
trustworthy. For the low-N / non-i.i.d. regime use
:class:`~deup.diagnostics.health.HealthIndex` instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from deup.core.grouping import Grouping

# Documented empirical reference points (orientation only; see module docstring).
REFERENCE_POINTS: dict[int, float] = {50: 0.55, 10_000: 0.955}


def _lag1_autocorr(values: npt.NDArray[Any]) -> float:
    """Lag-1 autocorrelation of a 1-D sequence (0.0 if degenerate)."""
    v = np.asarray(values, dtype=float)
    if v.shape[0] < 3:
        return 0.0
    v = v - v.mean()
    denom = float(np.dot(v, v))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(v[:-1], v[1:]) / denom)


def effective_sample_size(values: npt.ArrayLike, n: int | None = None) -> float:
    """Autocorrelation-discounted effective sample size.

    Uses the standard lag-1 AR(1) inflation factor
    ``N_eff = N * (1 - rho) / (1 + rho)`` where ``rho`` is the lag-1 autocorrelation.
    Independent data (``rho ~ 0``) gives ``N_eff ~ N``; strong positive dependence
    (``rho -> 1``) shrinks ``N_eff`` toward 1. The order of ``values`` matters: pass
    them in their natural (e.g. temporal) order.

    Parameters
    ----------
    values:
        The within-context per-item signal in natural order.
    n:
        Override the raw count (defaults to ``len(values)``).
    """
    v = np.asarray(values, dtype=float)
    raw_n = int(v.shape[0] if n is None else n)
    if raw_n <= 1:
        return float(raw_n)
    rho = _lag1_autocorr(v)
    rho = max(min(rho, 0.999), 0.0)  # negative autocorrelation does not reduce info
    n_eff = raw_n * (1.0 - rho) / (1.0 + rho)
    return float(max(1.0, min(n_eff, raw_n)))


@dataclass(frozen=True)
class AggregationVerdict:
    """Outcome of an aggregation-reliability check."""

    trustworthy: bool
    median_effective_n: float
    median_raw_n: float
    median_autocorr: float
    n_contexts: int
    reason: str


class AggregationReliability:
    """Estimate whether an aggregated ``mean(g)`` context signal is trustworthy.

    Parameters
    ----------
    min_effective_n:
        Effective-N threshold below which aggregation is flagged untrustworthy.
        Defaults to ``200`` -- comfortably above the ``N ~ 50`` regime that scored
        near-chance and well below the ``N ~ 10,000`` regime that saturated, while
        leaving headroom for the autocorrelation discount.
    max_autocorr:
        Median within-context lag-1 autocorrelation above which dependence is judged
        high enough to undermine the i.i.d. assumption.

    Notes
    -----
    The thresholds are conservative defaults derived from the empirical reference
    points (see :data:`REFERENCE_POINTS`); tune them for your domain. This guard is the
    explicit remedy for silently exposing ``context_uncertainty = mean(g)``.
    """

    def __init__(
        self,
        *,
        min_effective_n: float = 200.0,
        max_autocorr: float = 0.2,
    ) -> None:
        self.min_effective_n = min_effective_n
        self.max_autocorr = max_autocorr

    def analyze(
        self,
        g: npt.ArrayLike,
        groups: npt.ArrayLike,
    ) -> AggregationVerdict:
        """Compute per-context N_eff / autocorrelation and a trustworthiness verdict.

        Parameters
        ----------
        g:
            Per-item epistemic estimate ``g(x_i)``.
        groups:
            Per-item context label (e.g. date). Items within a group are assumed to be
            in natural (temporal) order.
        """
        g_arr = np.asarray(g, dtype=float)
        grouping = Grouping.from_labels(groups, g_arr.shape[0])

        raw_ns: list[int] = []
        eff_ns: list[float] = []
        autocorrs: list[float] = []
        for idx in grouping.indices():
            vals = g_arr[idx]
            raw_ns.append(int(vals.shape[0]))
            eff_ns.append(effective_sample_size(vals))
            autocorrs.append(abs(_lag1_autocorr(vals)))

        median_raw = float(np.median(raw_ns))
        median_eff = float(np.median(eff_ns))
        median_ac = float(np.median(autocorrs))

        enough_n = median_eff >= self.min_effective_n
        low_dependence = median_ac <= self.max_autocorr
        trustworthy = bool(enough_n and low_dependence)

        if trustworthy:
            reason = (
                f"aggregate trustworthy: median N_eff={median_eff:.0f} "
                f">= {self.min_effective_n:.0f} and median |autocorr|={median_ac:.2f} "
                f"<= {self.max_autocorr:.2f}"
            )
        else:
            parts = []
            if not enough_n:
                parts.append(f"median N_eff={median_eff:.0f} < {self.min_effective_n:.0f}")
            if not low_dependence:
                parts.append(f"median |autocorr|={median_ac:.2f} > {self.max_autocorr:.2f}")
            reason = (
                "aggregate NOT trustworthy (" + "; ".join(parts) + "); "
                "prefer a composite HealthIndex over raw mean(g)."
            )

        return AggregationVerdict(
            trustworthy=trustworthy,
            median_effective_n=median_eff,
            median_raw_n=median_raw,
            median_autocorr=median_ac,
            n_contexts=grouping.n_groups,
            reason=reason,
        )

    def aggregate(
        self,
        g: npt.ArrayLike,
        groups: npt.ArrayLike,
        *,
        warn: bool = True,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], AggregationVerdict]:
        """Return ``(context_labels, mean_g_per_context, verdict)``.

        Emits a :class:`UserWarning` when the aggregate is judged untrustworthy (unless
        ``warn=False``). This is the guarded alternative to a bare ``mean(g)`` API.
        """
        g_arr = np.asarray(g, dtype=float)
        grouping = Grouping.from_labels(groups, g_arr.shape[0])
        labels = grouping.labels
        means = np.array([g_arr[idx].mean() for idx in grouping.indices()], dtype=float)
        verdict = self.analyze(g, groups)
        if warn and not verdict.trustworthy:
            warnings.warn(verdict.reason, UserWarning, stacklevel=2)
        return labels, means, verdict


def should_trust_aggregate(
    g: npt.ArrayLike,
    groups: npt.ArrayLike,
    *,
    min_effective_n: float = 200.0,
    max_autocorr: float = 0.2,
) -> AggregationVerdict:
    """Convenience: return a trust verdict (+ reason) for ``mean(g)`` aggregation.

    Thin wrapper over :class:`AggregationReliability` for one-off checks.
    """
    return AggregationReliability(
        min_effective_n=min_effective_n, max_autocorr=max_autocorr
    ).analyze(g, groups)
