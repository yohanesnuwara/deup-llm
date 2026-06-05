"""Executable smoke tests mirroring docs/tutorials/*.md (P13 CI gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def test_tutorial_tabular_regression() -> None:
    """docs/tutorials/tabular-regression.md"""
    from deup import DEUPRegressor
    from deup.domains.tabular import TabularDEUP

    X, y = fetch_california_housing(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, _y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    model = DEUPRegressor(
        base_model=RandomForestRegressor(n_estimators=30, random_state=0),
        cv=3,
        random_state=0,
    )
    model.fit(X_tr, y_tr)
    pred, unc = model.predict(X_te, return_uncertainty=True)
    assert pred.shape[0] == X_te.shape[0]
    assert unc.shape == pred.shape
    assert np.all(unc >= 0)

    preset = TabularDEUP(cv=3, random_state=0)
    preset.fit(X_tr, y_tr)
    unc2 = preset.predict_epistemic(X_te)
    assert unc2.shape[0] == X_te.shape[0]


def test_tutorial_finance_ranking() -> None:
    """docs/tutorials/finance-ranking.md"""
    from deup.domains.finance import CrossSectionalDEUP

    rng = np.random.default_rng(0)
    n_dates, n_assets = 8, 12
    rows = []
    for d in range(n_dates):
        for a in range(n_assets):
            rows.append(
                {
                    "date": f"2020-01-{d + 1:02d}",
                    "asset": f"A{a}",
                    "score": float(rng.normal()),
                    "vol_20d": float(rng.uniform(0.01, 0.05)),
                    "market_vol_21d": float(rng.uniform(0.01, 0.04)),
                    "target_20d": float(rng.normal()),
                }
            )
    panel = pd.DataFrame(rows)
    tr, te = train_test_split(panel, test_size=0.3, random_state=0)
    cal, te = train_test_split(te, test_size=0.5, random_state=0)

    model = CrossSectionalDEUP(horizon=20, cv=3, embargo=1, random_state=0)
    model.fit(tr)
    pred, unc = model.predict(te, return_uncertainty=True)
    assert len(pred) == len(te)
    assert len(unc) == len(te)

    model.calibrate(cal, alpha=0.1)
    interval = model.predict_interval(te.head(5))
    assert len(interval.lower) == 5

    report = model.health_report(te)
    assert len(report.health) == te["date"].nunique()


def test_tutorial_classification_conformal() -> None:
    """docs/tutorials/classification-conformal.md"""
    from deup import DEUPClassifier

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, _y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)

    clf = DEUPClassifier(
        base_model=RandomForestClassifier(n_estimators=30, random_state=0),
        cv=3,
        random_state=0,
    )
    clf.fit(X_tr, y_tr)
    pred, unc = clf.predict(X_te, return_uncertainty=True)
    proba = clf.predict_proba(X_te)
    assert pred.shape[0] == len(X_te)
    assert unc.shape == pred.shape
    assert proba.shape[0] == len(X_te)

    clf.calibrate(X_cal, y_cal, method="normalized", alpha=0.1)
    result = clf.predict_interval(X_te[:10])
    assert len(result.lower) == 10


def test_tutorial_active_learning() -> None:
    """docs/tutorials/active-learning.md"""
    from deup import DEUPClassifier

    X, y = make_classification(n_samples=400, n_features=8, random_state=0)
    X_lab, X_pool, y_lab, _y_pool = train_test_split(X, y, test_size=0.6, random_state=0)

    model = DEUPClassifier(
        base_model=RandomForestClassifier(n_estimators=30, random_state=0),
        cv=3,
        random_state=0,
    )
    model.fit(X_lab, y_lab)
    idx = model.acquire(X_pool, k=5)
    assert idx.shape == (5,)
    idx2, unc = model.acquire(X_pool, k=5, return_uncertainty=True)
    assert idx2.shape == (5,)
    assert unc.shape == (5,)
