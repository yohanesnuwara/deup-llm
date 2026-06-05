"""Fast smoke test for benchmark helpers (not the full California run)."""

from __future__ import annotations

import numpy as np
from benchmarks.run_regression_benchmark import benchmark_deup, benchmark_ensemble
from sklearn.model_selection import train_test_split


def test_deup_benchmark_helper_on_tiny_data() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.0, -1.0, 0.5, 0.0]) + rng.normal(scale=0.3, size=200)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    out = benchmark_deup(X_tr, y_tr, X_te, y_te, seed=0)
    assert np.isfinite(out["spearman"])
    assert out["unc_mean"] >= 0.0


def test_ensemble_benchmark_helper_on_tiny_data() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.0, -1.0, 0.5, 0.0]) + rng.normal(scale=0.3, size=200)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    out = benchmark_ensemble(X_tr, y_tr, X_te, y_te, seed=0)
    assert np.isfinite(out["spearman"])
    assert out["unc_mean"] >= 0.0


def test_n_sweep_iid_row_smoke() -> None:
    from benchmarks.run_n_sweep import run_iid_n_sweep

    rows = run_iid_n_sweep(ns=[50], max_total_samples=5_000)
    assert len(rows) == 1
    assert float(rows[0]["auroc_agg_g"]) > 0.5
