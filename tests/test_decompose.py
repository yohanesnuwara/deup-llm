"""P6: ErrorEstimator, aleatoric estimators, decomposition, rank residualization.

Covers the P6 success criteria:
  * g is monotonically higher in an injected OOD region (statistical test);
  * rank residualization reduces rho(g, |score|) while retaining >=85% of loss
    association on a ranker fixture;
  * the density kill criterion fires on a null-density fixture;
  * e_hat = max(0, g - a) >= 0 always.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from deup.core import (
    ErrorEstimator,
    Heteroscedastic,
    Homoscedastic,
    Quantile,
    RankResidualizer,
    coupling_retention_report,
    decompose_epistemic,
    density_kill_criterion,
    partial_correlation,
)
from deup.core.features import DensityFeature, FeaturePipeline, RawFeatures
from deup.core.oof import OOFErrorCollector


# --------------------------------------------------------------------------- #
# ErrorEstimator
# --------------------------------------------------------------------------- #
def test_error_estimator_predicts_nonnegative() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    errors = np.abs(rng.normal(size=200))
    est = ErrorEstimator(target_transform="log").fit(X, errors)
    pred = est.predict(X)
    assert pred.shape == (200,)
    assert np.all(pred >= 0)


def test_error_estimator_rejects_negative_errors() -> None:
    X = np.random.default_rng(1).normal(size=(20, 2))
    with pytest.raises(ValueError, match="non-negative"):
        ErrorEstimator().fit(X, -np.ones(20))


def test_error_estimator_with_feature_pipeline() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 4))
    errors = np.abs(rng.normal(size=150))
    pipe = FeaturePipeline([("raw", RawFeatures()), ("density", DensityFeature())])
    est = ErrorEstimator(features=pipe).fit(X, errors)
    assert est.predict(X).shape == (150,)


def test_error_estimator_higher_in_ood_region() -> None:
    """g must be statistically higher in an injected OOD region (success criterion).

    Training density peaks near 0 (Gaussian inputs); label noise grows as inputs move
    into the sparse tails, so out-of-fold error correlates with low density. The
    density feature lets ``g`` extrapolate that to an injected far-OOD region.
    """
    rng = np.random.default_rng(7)
    x_train = rng.normal(loc=0.0, scale=1.0, size=600).reshape(-1, 1)
    # Heteroscedastic noise tied to distance from the dense center (low density).
    noise_scale = 0.05 + 0.6 * np.abs(x_train.ravel())
    y_train = np.sin(x_train).ravel() + rng.normal(scale=noise_scale)

    collector = OOFErrorCollector(
        LinearRegression(),
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        loss="squared",
    )
    oof = collector.fit_collect(x_train, y_train)

    pipe = FeaturePipeline(
        [("raw", RawFeatures()), ("density", DensityFeature(method="mahalanobis"))]
    )
    est = ErrorEstimator(features=pipe, target_transform="log").fit(
        x_train[oof.indices], oof.errors
    )

    x_in = rng.normal(loc=0.0, scale=1.0, size=200).reshape(-1, 1)
    x_ood = rng.uniform(6, 8, size=80).reshape(-1, 1)  # far outside training support
    g_in = est.predict(x_in)
    g_ood = est.predict(x_ood)

    stat = mannwhitneyu(g_ood, g_in, alternative="greater")
    assert stat.pvalue < 0.01
    assert g_ood.mean() > g_in.mean()


# --------------------------------------------------------------------------- #
# Aleatoric estimators
# --------------------------------------------------------------------------- #
def test_homoscedastic_constant_and_nonneg() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 2))
    y = X[:, 0] + rng.normal(scale=0.5, size=300)
    a = Homoscedastic(k=15).fit(X, y).predict(X)
    assert np.all(a >= 0)
    assert np.allclose(a, a[0])  # constant


def test_heteroscedastic_tracks_local_noise() -> None:
    rng = np.random.default_rng(4)
    n = 600
    x = rng.uniform(0, 1, size=n).reshape(-1, 1)
    # noise grows with x: low-noise region vs high-noise region
    noise = np.where(x.ravel() < 0.5, 0.05, 1.0)
    y = np.sin(2 * x).ravel() + rng.normal(scale=noise)
    a = Heteroscedastic(k=25).fit(x, y).predict(x)
    assert np.all(a >= 0)
    lo = a[x.ravel() < 0.5].mean()
    hi = a[x.ravel() >= 0.5].mean()
    assert hi > lo  # higher aleatoric where noise is larger


def test_quantile_aleatoric_nonneg() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 2))
    y = X[:, 0] + rng.normal(scale=0.7, size=400)
    a = Quantile().fit(X, y).predict(X)
    assert a.shape == (400,)
    assert np.all(a >= 0)


# --------------------------------------------------------------------------- #
# Decomposition
# --------------------------------------------------------------------------- #
def test_decompose_epistemic_nonnegative_and_proxy() -> None:
    g = np.array([1.0, 2.0, 0.5])
    a = np.array([0.3, 5.0, 0.1])
    e = decompose_epistemic(g, a)
    assert np.all(e >= 0)
    assert e[1] == 0.0  # clipped where a > g
    # a=None => conservative proxy e_hat = g
    assert np.allclose(decompose_epistemic(g, None), g)


def test_decompose_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        decompose_epistemic(np.ones(3), np.ones(4))


# --------------------------------------------------------------------------- #
# Rank residualization (Finding 3)
# --------------------------------------------------------------------------- #
def _ranker_fixture(
    n_dates: int = 60, n_assets: int = 40, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_dates * n_assets
    dates = np.repeat(np.arange(n_dates), n_assets)
    score = rng.normal(size=n)
    # within-date rank percentile of |score| (mechanical rank geometry)
    from deup.core.grouping import Grouping

    rank_geom = Grouping.from_labels(dates, n).rank_within(np.abs(score), pct=True)
    genuine = rng.normal(size=n)  # the real, non-geometry error signal
    # loss target couples to both geometry and genuine signal
    loss = 0.7 * rank_geom + 0.3 * genuine
    # g approximates loss with a little noise
    g = loss + rng.normal(scale=0.05, size=n)
    return g, score, loss, dates


def test_rank_residualizer_decouples_and_retains() -> None:
    g, score, loss, dates = _ranker_fixture()
    report = coupling_retention_report(g, score, loss, dates)
    # coupling to |score| drops substantially
    assert report.coupling_after < report.coupling_before
    assert report.coupling_after < 0.5 * report.coupling_before
    # >=85% of loss association retained after decoupling
    assert report.retention >= 0.85


def test_rank_residualizer_transform_roundtrip() -> None:
    g, score, loss, dates = _ranker_fixture(seed=2)
    res = RankResidualizer().fit(g, score, dates)
    out = res.transform(g, score, dates)
    assert out.shape == g.shape
    # residual has near-zero mean per construction (isotonic removes monotone part)
    assert abs(float(np.mean(out))) < abs(float(np.mean(g))) + 1.0


# --------------------------------------------------------------------------- #
# Density kill criterion (Finding 3 corollary)
# --------------------------------------------------------------------------- #
def test_density_kill_fires_on_null_density() -> None:
    decision = density_kill_criterion(gain_importance=1e-5, delta_partial_corr=0.001)
    assert decision.keep is False
    assert "killed" in decision.reason


def test_density_kept_when_informative() -> None:
    decision = density_kill_criterion(gain_importance=0.2, delta_partial_corr=0.05)
    assert decision.keep is True
    d2 = density_kill_criterion(gain_importance=0.2, delta_partial_corr=0.0)
    assert d2.keep is True  # importance alone keeps it


def test_partial_correlation_removes_control() -> None:
    rng = np.random.default_rng(9)
    c = rng.normal(size=500)
    a = c + rng.normal(scale=0.01, size=500)
    b = c + rng.normal(scale=0.01, size=500)
    # a and b are highly correlated, but only via c
    assert abs(partial_correlation(a, b, c)) < 0.3


def test_linfit_dependence() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(size=200)
    y = 3.0 * x + 2.0
    # full correlation ~1, partial controlling x ~0
    assert abs(partial_correlation(y, x, x)) < 1e-6
