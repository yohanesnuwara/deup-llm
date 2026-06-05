"""P9: aggregation-reliability + composite health index (Findings 1 & 2).

The two gate tests:
  * high-N i.i.d. fixture -> diagnostic says trustworthy AND mean(g) tracks context
    error (high rho);
  * low-N autocorrelated fixture -> diagnostic warns, mean(g) is near-chance, and
    HealthIndex recovers detection (materially higher AUROC than raw agg_g).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from deup.diagnostics import (
    AggregationReliability,
    HealthIndex,
    drift_psi,
    effective_sample_size,
    model_disagreement,
    realized_efficacy,
    should_trust_aggregate,
)


# --------------------------------------------------------------- effective N
def test_effective_n_independent_is_full() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(size=500)
    assert effective_sample_size(v) > 400  # ~ N for iid


def test_effective_n_autocorrelated_shrinks() -> None:
    rng = np.random.default_rng(0)
    n = 500
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = 0.95 * v[t - 1] + rng.normal(scale=0.3)
    assert effective_sample_size(v) < 100  # strong AR(1) collapses N_eff


# --------------------------------------------------------------- GATE TEST 1
def test_high_n_iid_aggregate_is_trustworthy_and_tracks_error() -> None:
    """Finding 1, high-N i.i.d. side: trustworthy verdict + mean(g) tracks error."""
    rng = np.random.default_rng(1)
    n_contexts = 40
    n_per = 3000  # high N per context
    groups = np.repeat(np.arange(n_contexts), n_per)

    # Each context has a latent difficulty; per-item g and error are noisy reads of it
    # with independent within-context noise (i.i.d.).
    difficulty = rng.uniform(0.5, 3.0, size=n_contexts)
    g = np.empty(n_contexts * n_per)
    err = np.empty(n_contexts * n_per)
    for c in range(n_contexts):
        sl = slice(c * n_per, (c + 1) * n_per)
        g[sl] = difficulty[c] + rng.normal(scale=0.5, size=n_per)
        err[sl] = difficulty[c] + rng.normal(scale=0.5, size=n_per)

    verdict = should_trust_aggregate(g, groups)
    assert verdict.trustworthy is True
    assert verdict.median_effective_n > 1000

    # mean(g) per context should track mean error per context (high rho)
    agg_g = np.array([g[groups == c].mean() for c in range(n_contexts)])
    agg_e = np.array([err[groups == c].mean() for c in range(n_contexts)])
    rho = spearmanr(agg_g, agg_e).statistic
    assert rho > 0.9


# --------------------------------------------------------------- GATE TEST 2
def _low_n_autocorr_fixture(seed: int = 2):
    """Low-N, temporally autocorrelated contexts; ~30% are 'bad' regimes.

    Returns per-item g, per-context bad-label, groups, and per-item arrays for the
    HealthIndex (loss, feature, disagreement) + a reference feature sample.
    """
    rng = np.random.default_rng(seed)
    n_contexts = 200
    n_per = 50  # low N per context (the regime that scored near-chance)

    # Persistent regime state (autocorrelated) drives "bad" contexts.
    state = np.zeros(n_contexts)
    for t in range(1, n_contexts):
        state[t] = 0.9 * state[t - 1] + rng.normal(scale=0.5)
    bad = (state > np.quantile(state, 0.7)).astype(int)

    groups = np.repeat(np.arange(n_contexts), n_per)
    g = np.empty(n_contexts * n_per)
    loss = np.empty(n_contexts * n_per)
    feature = np.empty(n_contexts * n_per)
    disagreement = np.empty(n_contexts * n_per)

    # Per-context idiosyncratic offset in g (regime-driven, NOT aligned with `bad`):
    # this is the component that does not average away and swamps the weak true signal,
    # the mechanism behind near-chance aggregated detection at low N (Finding 1).
    context_offset = rng.normal(scale=0.8, size=n_contexts)
    for c in range(n_contexts):
        sl = slice(c * n_per, (c + 1) * n_per)
        # g barely separates bad regimes at the individual level (weak + noisy),
        # and within-context errors are autocorrelated (AR(1)) -> mean(g) unreliable.
        base = 0.1 * bad[c] + context_offset[c]
        e = np.zeros(n_per)
        for t in range(1, n_per):
            e[t] = 0.8 * e[t - 1] + rng.normal(scale=1.0)
        g[sl] = base + 0.05 * e + rng.normal(scale=1.0, size=n_per)
        # Complementary signals that DO separate bad regimes (Finding 2):
        loss[sl] = 1.0 + 1.2 * bad[c] + rng.normal(scale=0.3, size=n_per)
        feature[sl] = bad[c] * 2.0 + rng.normal(scale=0.5, size=n_per)
        disagreement[sl] = 0.5 + 0.8 * bad[c] + rng.normal(scale=0.3, size=n_per)

    feature_reference = rng.normal(scale=0.5, size=5000)  # "healthy" baseline
    return g, bad, groups, loss, feature, feature_reference, disagreement


def test_low_n_autocorr_warns_and_health_recovers() -> None:
    g, bad, groups, loss, feature, feat_ref, disagreement = _low_n_autocorr_fixture()
    n_contexts = bad.shape[0]

    # (a) diagnostic warns and judges the aggregate untrustworthy
    reliab = AggregationReliability()
    verdict = reliab.analyze(g, groups)
    assert verdict.trustworthy is False
    with pytest.warns(UserWarning, match="NOT trustworthy"):
        _, agg_g, _ = reliab.aggregate(g, groups, warn=True)

    # (b) raw mean(g) is near-chance at separating bad regimes
    auroc_agg_g = roc_auc_score(bad, agg_g)
    assert auroc_agg_g < 0.65

    # (c) HealthIndex recovers detection (materially higher AUROC)
    health = HealthIndex(
        components=[
            ("realized_efficacy", realized_efficacy),
            ("drift_psi", drift_psi),
            ("model_disagreement", model_disagreement),
        ]
    )
    report = health.compute(
        groups,
        {
            "loss": loss,
            "feature": feature,
            "feature_reference": feat_ref,
            "disagreement": disagreement,
        },
    )
    assert report.health.shape == (n_contexts,)
    # health is LOW for bad regimes -> (1 - health) should rank bad contexts high
    auroc_health = roc_auc_score(bad, 1.0 - report.health)
    assert auroc_health > auroc_agg_g + 0.15
    assert auroc_health > 0.75


# --------------------------------------------------------------- components
def test_health_report_gate_and_verdict() -> None:
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(10), 20)
    loss = rng.normal(size=200)
    report = HealthIndex(components=[("realized_efficacy", realized_efficacy)]).compute(
        groups, {"loss": loss}
    )
    assert report.gate.shape == (10,)
    assert isinstance(report.verdict(report.labels[0]), bool)
    assert np.all((report.health >= 0) & (report.health <= 1))


def test_drift_psi_detects_shift() -> None:
    rng = np.random.default_rng(6)
    ref = rng.normal(size=2000)
    feature = np.concatenate([rng.normal(size=100), rng.normal(loc=3.0, size=100)])
    psi0 = drift_psi(np.arange(100), {"feature": feature, "feature_reference": ref})
    psi1 = drift_psi(np.arange(100, 200), {"feature": feature, "feature_reference": ref})
    assert psi1 > psi0  # shifted context has higher drift


def test_weights_length_validation() -> None:
    groups = np.repeat(np.arange(5), 10)
    loss = np.zeros(50)
    with pytest.raises(ValueError, match="weights length"):
        HealthIndex(
            components=[("realized_efficacy", realized_efficacy)],
            weights=[1.0, 2.0],
        ).compute(groups, {"loss": loss})


def test_aggregate_no_warn_when_trustworthy() -> None:
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(20), 2000)
    g = rng.normal(size=40000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # would raise if a warning fired
        _, _, verdict = AggregationReliability().aggregate(g, groups, warn=True)
    assert verdict.trustworthy is True
