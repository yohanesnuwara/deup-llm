"""Regression uncertainty benchmark — v0.1 killer benchmark.

Compares uncertainty quality on a held-out test set:
  1. DEUP (this library)
  2. Ensemble disagreement (bootstrap RandomForests)
  3. Split-conformal interval width (residual-based)

Metric: Spearman correlation between each method's uncertainty score and the
realized squared error on the test set. Higher is better.

Run:
    python benchmarks/run_regression_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Allow running without installing (dev checkout)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deup import DEUPRegressor  # noqa: E402


def _spearman_unc_vs_sqerr(unc: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    realized = (y_true - y_pred) ** 2
    return float(spearmanr(unc, realized).statistic)


def benchmark_deup(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    model = DEUPRegressor(
        base_model=RandomForestRegressor(n_estimators=80, random_state=seed),
        cv=5,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)
    pred, unc = model.predict(X_te, return_uncertainty=True)
    return {
        "spearman": _spearman_unc_vs_sqerr(unc, y_te, pred),
        "unc_mean": float(unc.mean()),
    }


def benchmark_ensemble(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    n_members: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    preds = []
    for k in range(n_members):
        idx = rng.choice(len(y_tr), size=len(y_tr), replace=True)
        m = RandomForestRegressor(n_estimators=80, random_state=seed + k)
        m.fit(X_tr[idx], y_tr[idx])
        preds.append(m.predict(X_te))
    stack = np.stack(preds, axis=0)
    pred = stack.mean(axis=0)
    unc = stack.var(axis=0)
    return {
        "spearman": _spearman_unc_vs_sqerr(unc, y_te, pred),
        "unc_mean": float(unc.mean()),
    }


def benchmark_conformal_residual(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Conformal-adjacent baseline: predict |residual| from a calibration split.

    Split conformal interval *width* is constant per run (not useful for ranking).
    Instead we fit a cal-set model for absolute residual magnitude — a standard
    split-conformal building block and a fairer uncertainty baseline.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed)
    base = RandomForestRegressor(n_estimators=80, random_state=seed)
    base.fit(X_fit, y_fit)
    cal_abs = np.abs(y_cal - base.predict(X_cal))
    resid_model = HistGradientBoostingRegressor(random_state=seed)
    resid_model.fit(X_cal, cal_abs)
    pred = base.predict(X_te)
    unc = np.clip(resid_model.predict(X_te), 0.0, None)
    return {
        "spearman": _spearman_unc_vs_sqerr(unc, y_te, pred),
        "unc_mean": float(unc.mean()),
    }


def n_sweep_teaser(*, seed: int = 0) -> list[dict[str, float]]:
    """Teaser for Finding 1: aggregate g tracks context error better as N grows."""
    from deup.splitters import KFold

    rng = np.random.default_rng(seed)
    rows = []
    for n_per_context in [10, 50, 200, 1000]:
        n_contexts = max(20, 8000 // n_per_context)
        X_parts, y_parts = [], []
        for _ in range(n_contexts):
            Xc = rng.normal(size=(n_per_context, 4))
            # heteroscedastic noise by |x0|
            noise = rng.normal(size=n_per_context) * (0.2 + 2.0 * np.abs(Xc[:, 0]))
            yc = Xc[:, 1] + noise
            X_parts.append(Xc)
            y_parts.append(yc)
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        groups = np.repeat(np.arange(n_contexts), n_per_context)

        deup = DEUPRegressor(
            base_model=RandomForestRegressor(n_estimators=40, random_state=seed),
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
            random_state=seed,
        )
        deup.fit(X, y)
        unc = deup.predict_epistemic(X)

        # context-level: mean g vs mean realized sq error
        ctx_g = []
        ctx_err = []
        pred = deup.base_model_.predict(X)
        sq = (y - pred) ** 2
        for g in range(n_contexts):
            mask = groups == g
            ctx_g.append(float(unc[mask].mean()))
            ctx_err.append(float(sq[mask].mean()))
        rho = float(spearmanr(ctx_g, ctx_err).statistic)
        rows.append({"n_per_context": n_per_context, "n_contexts": n_contexts, "agg_spearman": rho})
    return rows


def main() -> None:
    X, y = fetch_california_housing(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    results = {
        "dataset": "california_housing",
        "test_n": len(y_te),
        "methods": {
            "deup": benchmark_deup(X_tr, y_tr, X_te, y_te),
            "ensemble_disagreement": benchmark_ensemble(X_tr, y_tr, X_te, y_te),
            "conformal_residual": benchmark_conformal_residual(X_tr, y_tr, X_te, y_te),
        },
        "n_sweep_teaser": n_sweep_teaser(),
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "regression_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print("=== Regression uncertainty benchmark (California housing) ===")
    print(f"test n = {results['test_n']}")
    for name, metrics in results["methods"].items():
        print(f"  {name:24s} Spearman={metrics['spearman']:.4f}")
    print("\n=== N-sweep teaser (context-level agg Spearman) ===")
    for row in results["n_sweep_teaser"]:
        print(
            f"  N={row['n_per_context']:4d}/context  "
            f"contexts={row['n_contexts']:3d}  agg_rho={row['agg_spearman']:.4f}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
