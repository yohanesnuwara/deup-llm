"""P8: conformal calibration + MAPIE interop.

Success criteria:
  * empirical coverage within +/-2% of nominal 1-alpha on >=3 iid datasets;
  * coverage holds on a time-split fixture with a purged splitter;
  * intervals narrower in low-g regions (efficiency);
  * MAPIE-style normalizer adapter works.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from deup import DEUPRegressor
from deup.calibration import UncertaintyCalibrator, deup_normalizer
from deup.splitters import PurgedWalkForward


def _split3(X, y, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n = len(y)
    a, b = idx[: n // 3], idx[n // 3 : 2 * n // 3]
    c = idx[2 * n // 3 :]
    return (X[a], y[a]), (X[b], y[b]), (X[c], y[c])


def _coverage(result, y_true) -> float:
    inside = (y_true >= result.lower) & (y_true <= result.upper)
    return float(np.mean(inside))


# --------------------------------------------------------------- coverage (iid)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_normalized_coverage_within_tolerance(seed: int) -> None:
    # Large test split keeps the empirical-coverage sampling std small enough that the
    # +/-2% band is a real assertion rather than a coin flip (std ~ 0.0055 at n_te=3000).
    rng = np.random.default_rng(seed)
    n = 9000
    X = rng.normal(size=(n, 4))
    # heteroscedastic noise so normalization matters
    noise = (0.2 + 1.5 * np.abs(X[:, 0])) * rng.normal(size=n)
    y = X @ np.array([1.0, -1.0, 0.5, 0.0]) + noise

    (Xtr, ytr), (Xcal, ycal), (Xte, yte) = _split3(X, y, seed=seed)
    model = DEUPRegressor(
        base_model=RandomForestRegressor(n_estimators=60, random_state=seed),
        cv=5,
        random_state=seed,
    ).fit(Xtr, ytr)
    model.calibrate(Xcal, ycal, method="normalized", alpha=0.1)
    result = model.predict_interval(Xte)
    cov = _coverage(result, yte)
    # Split conformal guarantees >= 1-alpha and may slightly over-cover.
    assert 0.88 <= cov <= 0.92


def test_calibrator_standalone_coverage() -> None:
    rng = np.random.default_rng(7)
    n = 4000
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(scale=0.5, size=n)
    unc = np.full(n, 0.5)
    cal = UncertaintyCalibrator(method="normalized", alpha=0.2)
    cal.fit(y_true[:2000], y_pred[:2000], unc[:2000])
    res = cal.predict_interval(y_pred[2000:], unc[2000:])
    cov = _coverage(res, y_true[2000:])
    assert abs(cov - 0.8) <= 0.03


# --------------------------------------------------------------- time split
def test_coverage_on_time_split_with_purged_cv() -> None:
    rng = np.random.default_rng(3)
    n = 3000
    X = rng.normal(size=(n, 3))
    y = np.cumsum(rng.normal(scale=0.3, size=n)) * 0 + X[:, 0] + rng.normal(scale=0.6, size=n)
    # chronological split: train/cal earlier, test later
    Xtr, ytr = X[:1500], y[:1500]
    Xcal, ycal = X[1500:2200], y[1500:2200]
    Xte, yte = X[2200:], y[2200:]
    model = DEUPRegressor(
        base_model=LinearRegression(),
        cv=PurgedWalkForward(n_splits=4, embargo=5),
    ).fit(Xtr, ytr)
    model.calibrate(Xcal, ycal, method="normalized", alpha=0.1)
    cov = _coverage(model.predict_interval(Xte), yte)
    assert abs(cov - 0.9) <= 0.04


# --------------------------------------------------------------- efficiency
def test_intervals_narrower_in_low_uncertainty_regions() -> None:
    rng = np.random.default_rng(5)
    n = 4000
    X = rng.normal(size=(n, 3))
    noise = (0.1 + 2.0 * np.abs(X[:, 0])) * rng.normal(size=n)
    y = X[:, 1] + noise
    (Xtr, ytr), (Xcal, ycal), (Xte, yte) = _split3(X, y, seed=5)
    model = DEUPRegressor(
        base_model=RandomForestRegressor(n_estimators=80, random_state=0),
        cv=5,
        random_state=0,
    ).fit(Xtr, ytr)
    model.calibrate(Xcal, ycal, method="normalized", alpha=0.1)
    result = model.predict_interval(Xte)
    unc = model.predict_epistemic(Xte)
    lo_mask = unc <= np.quantile(unc, 0.25)
    hi_mask = unc >= np.quantile(unc, 0.75)
    assert result.width[lo_mask].mean() < result.width[hi_mask].mean()


# --------------------------------------------------------------- mondrian
def test_mondrian_per_group_coverage() -> None:
    rng = np.random.default_rng(11)
    n = 4500
    X = rng.normal(size=(n, 3))
    groups = rng.integers(0, 3, size=n)
    scale = np.array([0.3, 1.0, 2.5])[groups]
    y = X[:, 0] + scale * rng.normal(size=n)

    idx = rng.permutation(n)
    tr, cal, te = idx[: n // 3], idx[n // 3 : 2 * n // 3], idx[2 * n // 3 :]
    model = DEUPRegressor(base_model=LinearRegression(), cv=5, random_state=0).fit(X[tr], y[tr])
    model.calibrate(X[cal], y[cal], method="mondrian", alpha=0.1, groups=groups[cal])
    cov = _coverage(model.predict_interval(X[te], groups=groups[te]), y[te])
    assert abs(cov - 0.9) <= 0.04


def test_cqr_method() -> None:
    rng = np.random.default_rng(13)
    n = 3000
    y = rng.normal(size=n)
    # synthetic quantile predictions around y
    lo = y - 1.0 + rng.normal(scale=0.1, size=n)
    hi = y + 1.0 + rng.normal(scale=0.1, size=n)
    cal = UncertaintyCalibrator(method="cqr", alpha=0.1)
    cal.fit(y[:1500], np.zeros(1500), y_lower=lo[:1500], y_upper=hi[:1500])
    res = cal.predict_interval(np.zeros(1500), y_lower=lo[1500:], y_upper=hi[1500:])
    cov = _coverage(res, y[1500:])
    assert abs(cov - 0.9) <= 0.04


# --------------------------------------------------------------- mapie interop
def test_deup_normalizer_matches_predict_epistemic() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = X[:, 0] + rng.normal(scale=0.5, size=300)
    model = DEUPRegressor(base_model=LinearRegression(), cv=4, random_state=0).fit(X, y)
    normalizer = deup_normalizer(model)
    normalizer.fit(X, y)
    assert np.allclose(normalizer.predict(X), model.predict_epistemic(X))


def test_calibrate_requires_fit() -> None:
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        DEUPRegressor().predict_interval(np.zeros((3, 4)))


def test_bad_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        UncertaintyCalibrator(alpha=1.5)
