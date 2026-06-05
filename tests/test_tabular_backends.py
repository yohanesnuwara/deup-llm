"""P16: TabularDEUP gradient-boosting backends (LightGBM, XGBoost, CatBoost)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deup.domains.tabular import TabularDEUP, default_tabular_base_model


def _california_subset(
    n: int = 800, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = fetch_california_housing(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X, y = X[:n], y[:n]
    return train_test_split(X, y, test_size=0.25, random_state=seed)


@pytest.mark.parametrize("backend", ["sklearn", "lgbm", "xgb", "catboost"])
def test_tabular_backend_fit_predict(backend: str) -> None:
    if backend == "lgbm":
        pytest.importorskip("lightgbm")
    elif backend == "xgb":
        pytest.importorskip("xgboost")
    elif backend == "catboost":
        pytest.importorskip("catboost")

    X_tr, X_te, y_tr, y_te = _california_subset()
    model = TabularDEUP(backend=backend, cv=3, random_state=0).fit(X_tr, y_tr)
    unc = model.predict_epistemic(X_te)
    assert unc.shape == (len(y_te),)
    assert np.all(unc >= 0.0)
    assert np.all(np.isfinite(unc))
    assert model.backend == backend


@pytest.mark.parametrize("backend", ["lgbm", "xgb", "catboost"])
def test_tabular_backend_spearman_ranks_errors(backend: str) -> None:
    """Epistemic scores correlate with realized squared error (smoke quality gate)."""
    if backend == "lgbm":
        pytest.importorskip("lightgbm")
    elif backend == "xgb":
        pytest.importorskip("xgboost")
    else:
        pytest.importorskip("catboost")

    X_tr, X_te, y_tr, y_te = _california_subset(n=2000)
    model = TabularDEUP(backend=backend, cv=5, random_state=0).fit(X_tr, y_tr)
    pred, unc = model.predict(X_te, return_uncertainty=True)
    sq = (y_te - pred) ** 2
    rho_deup, _ = spearmanr(unc, sq)
    assert float(rho_deup) > 0.15


def test_default_tabular_base_model_sklearn() -> None:
    m = default_tabular_base_model("sklearn", task="regression", random_state=0)
    assert m is not None
