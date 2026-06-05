"""Composite context-reliability health index (Finding 2).

When aggregated raw ``mean(g)`` fails in the low-N / non-i.i.d. regime (Finding 1), a
composite index that fuses *complementary* signals recovers context-level detection
(Sanderink, 2026):

    H(c) = f( realized_efficacy(c), feature_drift(c), model_disagreement(c) )

Empirically the composite materially outperformed raw ``agg_g`` for context-level
"is this regime bad?" detection. ``HealthIndex`` is general (not finance-only): it takes
a list of pluggable component callables, each producing one scalar per context, and
combines their (sign-aware, normalized) values into a single monitored health score in
``[0, 1]`` (higher = healthier), with an optional gating threshold.

Three ready-made components are provided; users can add their own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from deup.core.grouping import Grouping

# A component maps (context indices, context arrays-dict) -> scalar for that context.
ComponentFn = Callable[[npt.NDArray[Any], dict[str, npt.NDArray[Any]]], float]


def _zscore(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    v = np.asarray(values, dtype=float)
    sd = float(v.std())
    if sd <= 0.0:
        return np.zeros_like(v)
    return np.asarray((v - v.mean()) / sd, dtype=float)


def _minmax(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    v = np.asarray(values, dtype=float)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


# --------------------------------------------------------------------------- #
# Component signals (each returns a per-context scalar where HIGHER = WORSE)
# --------------------------------------------------------------------------- #
def realized_efficacy(idx: npt.NDArray[Any], arrays: dict[str, npt.NDArray[Any]]) -> float:
    """Mean realized loss in the context (higher = worse). Requires ``arrays['loss']``."""
    loss = arrays["loss"][idx]
    return float(np.mean(loss))


def model_disagreement(idx: npt.NDArray[Any], arrays: dict[str, npt.NDArray[Any]]) -> float:
    """Mean ensemble/model disagreement in the context. Requires ``arrays['disagreement']``."""
    dis = arrays["disagreement"][idx]
    return float(np.mean(dis))


def drift_psi(
    idx: npt.NDArray[Any],
    arrays: dict[str, npt.NDArray[Any]],
    *,
    n_bins: int = 10,
) -> float:
    """Population Stability Index of the context feature vs. a reference distribution.

    Requires ``arrays['feature']`` (per-item scalar feature) and
    ``arrays['feature_reference']`` (1-D reference sample). Higher PSI = more drift.
    """
    feat = arrays["feature"][idx]
    ref = arrays["feature_reference"]
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(feat, bins=edges)
    eps = 1e-6
    ref_frac = ref_hist / max(ref_hist.sum(), 1) + eps
    cur_frac = cur_hist / max(cur_hist.sum(), 1) + eps
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


@dataclass(frozen=True)
class HealthReport:
    """Per-context health scores and gating verdicts."""

    labels: npt.NDArray[Any]
    health: npt.NDArray[Any]  # in [0, 1], higher = healthier
    components: dict[str, npt.NDArray[Any]]  # raw per-context component values
    threshold: float
    gate: npt.NDArray[Any]  # bool, True = trust/trade this context

    def verdict(self, label: Any) -> bool:
        """Gate decision for a single context label."""
        i = int(np.flatnonzero(self.labels == label)[0])
        return bool(self.gate[i])


class HealthIndex:
    """Fuse complementary component signals into one context-reliability scalar.

    Parameters
    ----------
    components:
        List of ``(name, fn)`` pairs. Each ``fn(idx, arrays)`` returns one scalar per
        context where **higher = worse** (more unhealthy). Defaults to the three
        signals from Finding 2 (realized loss, drift PSI, model disagreement); supply
        your own to extend or replace them.
    weights:
        Optional per-component weights (defaults to equal). Length must match
        ``components``.
    threshold:
        Health-score gating threshold in ``[0, 1]``; contexts at or above it are
        "trustworthy / trade". Default ``0.5``.

    Notes
    -----
    Component values are z-scored across contexts (so heterogeneous scales combine
    sensibly), summed with weights into a "badness" score, then mapped to a ``[0, 1]``
    health score via min-max with health = 1 - normalized_badness. This is the
    low-N/non-i.i.d. remedy and is intended to stay **off** the high-N i.i.d. default
    path (where individual-level ``g`` already saturates).
    """

    def __init__(
        self,
        components: list[tuple[str, ComponentFn]] | None = None,
        *,
        weights: npt.ArrayLike | None = None,
        threshold: float = 0.5,
    ) -> None:
        if components is None:
            components = [
                ("realized_efficacy", realized_efficacy),
                ("drift_psi", drift_psi),
                ("model_disagreement", model_disagreement),
            ]
        self.components = components
        self.weights = weights
        self.threshold = threshold

    def compute(
        self,
        groups: npt.ArrayLike,
        arrays: dict[str, npt.ArrayLike],
    ) -> HealthReport:
        """Compute per-context health from the provided per-item ``arrays``.

        Parameters
        ----------
        groups:
            Per-item context labels.
        arrays:
            Dict of per-item arrays needed by the components (e.g. ``loss``,
            ``feature`` + ``feature_reference``, ``disagreement``). Reference arrays
            (keys ending in ``_reference``) are passed through unindexed.
        """
        n = np.asarray(groups).shape[0]
        grouping = Grouping.from_labels(groups, n)
        arr: dict[str, npt.NDArray[Any]] = {
            k: np.asarray(v, dtype=float) for k, v in arrays.items()
        }
        indices = grouping.indices()

        raw: dict[str, npt.NDArray[Any]] = {}
        for name, fn in self.components:
            raw[name] = np.array([fn(idx, arr) for idx in indices], dtype=float)

        w: npt.NDArray[Any]
        if self.weights is None:
            w = np.ones(len(self.components), dtype=float)
        else:
            w = np.asarray(self.weights, dtype=float)
            if w.shape[0] != len(self.components):
                raise ValueError("weights length must match number of components")
        w = w / w.sum()

        # Higher component value = worse; z-score then weighted-sum into "badness".
        badness = np.zeros(grouping.n_groups, dtype=float)
        for (name, _), wi in zip(self.components, w, strict=True):
            badness += wi * _zscore(raw[name])
        health = 1.0 - _minmax(badness)
        gate = health >= self.threshold

        return HealthReport(
            labels=grouping.labels,
            health=health,
            components=raw,
            threshold=self.threshold,
            gate=gate,
        )
