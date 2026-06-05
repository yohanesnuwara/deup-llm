"""P7: DEUPRegressor, DEUPClassifier, DEUPRanker + acquire."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression

from deup import DEUPClassifier, DEUPRanker, DEUPRegressor
from deup.core import Homoscedastic
from deup.splitters import PurgedWalkForward


def _make_regression(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(scale=0.5, size=n)
    return X, y


def _make_classification(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def _make_ranking(
    n_dates: int = 20, n_assets: int = 30, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_dates * n_assets
    dates = np.repeat(np.arange(n_dates), n_assets)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + rng.normal(scale=0.3, size=n)
    return X, y, dates


# ------------------------------------------------------------------ regression
def test_regressor_fit_predict_shapes() -> None:
    X, y = _make_regression()
    model = DEUPRegressor(base_model=LinearRegression(), cv=4, random_state=0).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(y),)


def test_regressor_return_uncertainty_tuple() -> None:
    X, y = _make_regression()
    model = DEUPRegressor(base_model=RandomForestRegressor(n_estimators=50), random_state=0)
    model.fit(X, y)
    pred, unc = model.predict(X, return_uncertainty=True)
    assert pred.shape == unc.shape == (len(y),)
    assert np.all(unc >= 0.0)


def test_regressor_with_aleatoric_decomposition() -> None:
    X, y = _make_regression(n=300)
    model = DEUPRegressor(
        base_model=LinearRegression(),
        aleatoric=Homoscedastic(k=10),
        cv=4,
        random_state=0,
    ).fit(X, y)
    unc = model.predict_epistemic(X)
    assert np.all(unc >= 0.0)


def test_regressor_acquire_returns_highest_epistemic() -> None:
    rng = np.random.default_rng(42)
    X_train, y_train = _make_regression(n=500)
    model = DEUPRegressor(base_model=LinearRegression(), cv=5, random_state=0).fit(X_train, y_train)

    # Pool: in-distribution + far OOD points (higher expected error)
    X_pool = np.vstack([rng.normal(size=(50, 4)), rng.uniform(8, 10, size=(10, 4))])
    unc = model.predict_epistemic(X_pool)
    idx = model.acquire(X_pool, k=5)
    assert len(idx) == 5
    # Acquired points should have higher uncertainty than pool median
    assert unc[idx].mean() >= np.median(unc)


def test_regressor_sklearn_clone_and_params() -> None:
    model = DEUPRegressor(base_model=LinearRegression(), cv=3, log_target=False)
    cloned = clone(model)
    assert cloned.get_params()["cv"] == 3
    assert cloned.get_params()["log_target"] is False
    assert isinstance(cloned.get_params()["base_model"], LinearRegression)


def test_regressor_walk_forward() -> None:
    X, y = _make_regression(n=300)
    model = DEUPRegressor(
        base_model=LinearRegression(),
        cv=PurgedWalkForward(n_splits=4, embargo=2),
    ).fit(X, y)
    pred, unc = model.predict(X[:5], return_uncertainty=True)
    assert pred.shape == (5,)
    assert np.all(unc >= 0.0)


def test_regressor_uncertainty_tracks_noise() -> None:
    rng = np.random.default_rng(1)
    n = 1500
    X = rng.normal(size=(n, 3))
    noise = rng.normal(size=n) * (0.1 + 2.0 * np.abs(X[:, 0]))
    y = X[:, 1] + noise
    model = DEUPRegressor(
        base_model=RandomForestRegressor(n_estimators=80, random_state=0),
        cv=5,
        random_state=0,
    ).fit(X[:1000], y[:1000])
    pred, unc = model.predict(X[1000:], return_uncertainty=True)
    rho = spearmanr(unc, (y[1000:] - pred) ** 2).statistic
    assert rho > 0.2


# ------------------------------------------------------------------ classification
def test_classifier_end_to_end() -> None:
    X, y = _make_classification()
    model = DEUPClassifier(
        base_model=LogisticRegression(max_iter=500),
        cv=4,
        random_state=0,
    ).fit(X, y)
    pred = model.predict(X[:20])
    proba = model.predict_proba(X[:20])
    _, unc = model.predict(X[:20], return_uncertainty=True)
    assert pred.shape == (20,)
    assert proba.shape == (20, 2)
    assert unc.shape == (20,)
    assert np.all(unc >= 0.0)


def test_classifier_brier_loss() -> None:
    X, y = _make_classification(n=300)
    model = DEUPClassifier(
        base_model=RandomForestClassifier(n_estimators=30, random_state=0),
        loss="brier",
        cv=4,
        random_state=0,
    ).fit(X, y)
    unc = model.predict_epistemic(X)
    assert np.all(unc >= 0.0)


def test_classifier_acquire() -> None:
    X, y = _make_classification(n=500)
    model = DEUPClassifier(base_model=LogisticRegression(max_iter=500), cv=4, random_state=0).fit(
        X, y
    )
    idx = model.acquire(X, k=10)
    unc = model.predict_epistemic(X)
    assert len(idx) == 10
    assert unc[idx].min() >= np.partition(unc, -10)[-10]


# ------------------------------------------------------------------ ranking
def test_ranker_requires_groups_at_fit() -> None:
    X, y, _ = _make_ranking()
    with pytest.raises(ValueError, match="groups"):
        DEUPRanker(base_model=LinearRegression(), cv=3).fit(X, y)


def test_ranker_end_to_end() -> None:
    X, y, dates = _make_ranking()
    model = DEUPRanker(
        base_model=LinearRegression(),
        cv=PurgedWalkForward(n_splits=3, embargo=0),
        random_state=0,
    ).fit(X, y, groups=dates)
    pred, unc = model.predict(X[:40], return_uncertainty=True, groups=dates[:40])
    assert pred.shape == (40,)
    assert np.all(unc >= 0.0)


def test_ranker_uses_residualized_signal() -> None:
    """Ranker with residualize_rank=True decouples from |score| geometry."""
    X, y, dates = _make_ranking(n_dates=40, n_assets=25, seed=1)
    kwargs = dict(
        base_model=LinearRegression(),
        cv=PurgedWalkForward(n_splits=3, embargo=0),
    )
    model_on = DEUPRanker(**kwargs, residualize_rank=True).fit(X, y, groups=dates)
    model_off = DEUPRanker(**kwargs, residualize_rank=False).fit(X, y, groups=dates)

    scores = np.asarray(model_on.base_model_.predict(X), dtype=float)
    abs_score = np.abs(scores)
    unc_on = model_on.predict_epistemic(X, groups=dates)
    unc_off = model_off.predict_epistemic(X, groups=dates)

    coupling_on = abs(float(spearmanr(unc_on, abs_score).statistic))
    coupling_off = abs(float(spearmanr(unc_off, abs_score).statistic))
    assert coupling_on < coupling_off
    assert model_on._residualizer_ is not None
    assert model_off._residualizer_ is None


def test_ranker_without_residualization() -> None:
    X, y, dates = _make_ranking()
    model = DEUPRanker(
        base_model=LinearRegression(),
        cv=3,
        residualize_rank=False,
    ).fit(X, y, groups=dates)
    assert model._residualizer_ is None
    unc = model.predict_epistemic(X, groups=dates)
    assert np.all(unc >= 0.0)


def test_ranker_acquire() -> None:
    X, y, dates = _make_ranking()
    model = DEUPRanker(base_model=LinearRegression(), cv=3).fit(X, y, groups=dates)
    idx = model.acquire(X, k=5, groups=dates)
    assert len(idx) == 5


def test_predict_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        DEUPRegressor().predict(np.zeros((3, 4)))
    with pytest.raises(NotFittedError):
        DEUPClassifier().predict(np.zeros((3, 4)))
    with pytest.raises(NotFittedError):
        DEUPRanker().predict(np.zeros((3, 4)))


def test_log_vs_raw_target_nonnegative() -> None:
    X, y = _make_regression(n=200)
    for transform in ("log", "asinh", "none"):
        m = DEUPRegressor(
            base_model=LinearRegression(),
            cv=4,
            target_transform=transform,
            random_state=0,
        ).fit(X, y)
        assert np.all(m.predict_epistemic(X) >= 0.0)
