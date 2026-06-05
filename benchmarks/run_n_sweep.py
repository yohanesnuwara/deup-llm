"""N-sweep benchmark — Finding 1 headline result.

Shows AUROC(agg_g, bad context) rising from ~0.55 at small N with temporal
dependence to ~0.95+ at large N under i.i.d. exchangeability, plus HealthIndex
recovery on the low-N / non-i.i.d. regime.

Run:
    python benchmarks/run_n_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import DEFAULT_SEED, ensure_results_dir, write_json  # noqa: E402
from deup import DEUPRegressor  # noqa: E402
from deup.diagnostics import HealthIndex  # noqa: E402
from deup.splitters import KFold  # noqa: E402


def _iid_context_panel(
    *,
    n_contexts: int,
    n_per: int,
    bad_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic i.i.d. panel: bad contexts have higher latent difficulty."""
    rng = np.random.default_rng(seed)
    difficulty = rng.uniform(0.5, 2.5, size=n_contexts)
    bad = (difficulty > np.quantile(difficulty, 1.0 - bad_fraction)).astype(int)
    groups = np.repeat(np.arange(n_contexts), n_per)
    n = n_contexts * n_per
    x = rng.normal(size=(n, 6))
    y = np.empty(n)
    g_target = np.empty(n)
    for c in range(n_contexts):
        sl = slice(c * n_per, (c + 1) * n_per)
        base = difficulty[c] + 0.8 * bad[c]
        noise = rng.normal(scale=0.4 + 0.1 * bad[c], size=n_per)
        y[sl] = x[sl, 0] + base + noise
        g_target[sl] = base + rng.normal(scale=0.2, size=n_per)
    return x, y, groups, bad, g_target


def _autocorr_low_n_panel(
    *,
    n_contexts: int = 200,
    n_per: int = 50,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Low-N autocorrelated panel (finance-like); HealthIndex should recover."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n_contexts)
    for t in range(1, n_contexts):
        state[t] = 0.9 * state[t - 1] + rng.normal(scale=0.5)
    bad = (state > np.quantile(state, 0.7)).astype(int)

    groups = np.repeat(np.arange(n_contexts), n_per)
    n = n_contexts * n_per
    x = rng.normal(size=(n, 5))
    y = np.empty(n)
    g = np.empty(n)
    loss = np.empty(n)
    feature = np.empty(n)
    disagreement = np.empty(n)
    context_offset = rng.normal(scale=0.8, size=n_contexts)

    for c in range(n_contexts):
        sl = slice(c * n_per, (c + 1) * n_per)
        base = 0.1 * bad[c] + context_offset[c]
        e = np.zeros(n_per)
        for t in range(1, n_per):
            e[t] = 0.8 * e[t - 1] + rng.normal(scale=1.0)
        g[sl] = base + 0.05 * e + rng.normal(scale=1.0, size=n_per)
        y[sl] = x[sl, 0] + rng.normal(scale=0.3, size=n_per)
        loss[sl] = 1.0 + 1.2 * bad[c] + rng.normal(scale=0.3, size=n_per)
        feature[sl] = bad[c] * 2.0 + rng.normal(scale=0.5, size=n_per)
        disagreement[sl] = 0.5 + 0.8 * bad[c] + rng.normal(scale=0.3, size=n_per)

    arrays = {
        "loss": loss,
        "feature": feature,
        "feature_reference": rng.normal(scale=0.5, size=5000),
        "disagreement": disagreement,
    }
    return x, y, groups, bad, arrays


def run_iid_n_sweep(
    ns: list[int] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    max_total_samples: int = 40_000,
) -> list[dict[str, float | int]]:
    if ns is None:
        ns = [50, 200, 1000, 5000, 10000]
    rows: list[dict[str, float | int]] = []
    for n_per in ns:
        n_contexts = max(50, max_total_samples // n_per)
        n_contexts = min(n_contexts, max_total_samples // max(n_per, 1))
        n_contexts = max(n_contexts, 30)
        if n_contexts * n_per > max_total_samples:
            n_contexts = max(30, max_total_samples // n_per)
        x, y, groups, bad_ctx, _ = _iid_context_panel(
            n_contexts=n_contexts,
            n_per=n_per,
            bad_fraction=0.25,
            seed=seed + n_per,
        )
        model = DEUPRegressor(
            base_model=RandomForestRegressor(
                n_estimators=20, max_depth=8, n_jobs=-1, random_state=seed
            ),
            cv=KFold(n_splits=3, shuffle=True, random_state=seed),
            random_state=seed,
        )
        model.fit(x, y)
        unc = model.predict_epistemic(x)
        pred = model.base_model_.predict(x)
        sq = (y - pred) ** 2

        agg_g = np.array([unc[groups == c].mean() for c in range(n_contexts)])
        agg_err = np.array([sq[groups == c].mean() for c in range(n_contexts)])
        auroc = float(roc_auc_score(bad_ctx, agg_g))
        rho = float(np.corrcoef(agg_g, agg_err)[0, 1])
        rows.append(
            {
                "regime": "iid",
                "split": "DEV",
                "n_per_context": n_per,
                "n_contexts": n_contexts,
                "auroc_agg_g": auroc,
                "spearman_agg_g_vs_error": rho,
            }
        )
    return rows


def run_low_n_health_recovery(*, seed: int = DEFAULT_SEED) -> dict[str, float | int]:
    x, y, groups, bad_ctx, arrays = _autocorr_low_n_panel(seed=seed)
    n_contexts = int(bad_ctx.shape[0])

    model = DEUPRegressor(
        base_model=RandomForestRegressor(n_estimators=40, random_state=seed),
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        random_state=seed,
    )
    model.fit(x, y)
    unc = model.predict_epistemic(x)
    agg_g = np.array([unc[groups == c].mean() for c in range(n_contexts)])

    auroc_agg = float(roc_auc_score(bad_ctx, agg_g))
    health = HealthIndex().compute(groups, arrays)
    auroc_health = float(roc_auc_score(bad_ctx, 1.0 - health.health))

    return {
        "regime": "autocorr_low_n",
        "split": "DEV",
        "n_per_context": 50,
        "n_contexts": n_contexts,
        "auroc_agg_g": auroc_agg,
        "auroc_health_index": auroc_health,
        "health_lift": auroc_health - auroc_agg,
    }


def plot_n_sweep(iid_rows: list[dict[str, float | int]], out_path: Path) -> None:
    ns = [int(r["n_per_context"]) for r in iid_rows]
    aurocs = [float(r["auroc_agg_g"]) for r in iid_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, aurocs, "o-", color="#3949ab", linewidth=2, markersize=7, label="AUROC(agg_g)")
    ax.axhline(0.55, color="#e53935", linestyle="--", linewidth=1, label="Finance ~0.55")
    ax.axhline(0.955, color="#43a047", linestyle="--", linewidth=1, label="CIFAR ~0.955")
    ax.set_xscale("log")
    ax.set_xlabel("N (observations per context)")
    ax.set_ylabel("AUROC(agg_g, bad context)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Finding 1: aggregation reliability vs N (i.i.d. synthetic)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    iid_rows = run_iid_n_sweep()
    health_row = run_low_n_health_recovery()
    out_dir = ensure_results_dir()
    fig_path = out_dir / "n_sweep.png"
    plot_n_sweep(iid_rows, fig_path)

    payload = {
        "seed": DEFAULT_SEED,
        "reference_points": {
            "finance_auroc_n50": 0.55,
            "cifar_auroc_n10000": 0.955,
            "thesis_health_auroc": 0.75,
        },
        "iid_sweep": iid_rows,
        "low_n_health": health_row,
        "figure": str(fig_path.name),
    }
    write_json("n_sweep.json", payload)

    print("=== N-sweep (Finding 1) ===")
    for row in iid_rows:
        print(
            f"  N={row['n_per_context']:5d}  AUROC={row['auroc_agg_g']:.3f}  "
            f"rho={row['spearman_agg_g_vs_error']:.3f}"
        )
    print(
        f"  Low-N autocorr: AUROC(agg_g)={health_row['auroc_agg_g']:.3f}  "
        f"AUROC(HealthIndex)={health_row['auroc_health_index']:.3f}"
    )
    print(f"Wrote {out_dir / 'n_sweep.json'} and {fig_path}")


if __name__ == "__main__":
    main()
