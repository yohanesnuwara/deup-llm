"""Reliability diagnostics for *aggregated* DEUP signals.

DEUP's ``g(x)`` is reliable at the **individual** level across domains, but the
reliability of an *aggregated* context signal ``mean_i g(x_i)`` is governed by the
number of observations per context (N) and whether they are exchangeable
(i.i.d.) vs. carrying temporal-regime dependence (Sanderink, 2026).

This subpackage operationalizes that finding:

- :class:`~deup.diagnostics.aggregation.AggregationReliability` — estimates effective
  N and within-context dependence, and warns when ``mean(g)`` is not trustworthy.
- :class:`~deup.diagnostics.health.HealthIndex` — a composite context-reliability
  scalar that fuses complementary signals (realized loss, feature drift, model
  disagreement) for the low-N / non-i.i.d. regime where raw ``mean(g)`` fails.
"""

from __future__ import annotations

from deup.diagnostics.aggregation import (
    AggregationReliability,
    AggregationVerdict,
    effective_sample_size,
    should_trust_aggregate,
)
from deup.diagnostics.health import (
    HealthIndex,
    HealthReport,
    drift_psi,
    model_disagreement,
    realized_efficacy,
)

__all__ = [
    "AggregationReliability",
    "AggregationVerdict",
    "HealthIndex",
    "HealthReport",
    "drift_psi",
    "effective_sample_size",
    "model_disagreement",
    "realized_efficacy",
    "should_trust_aggregate",
]
