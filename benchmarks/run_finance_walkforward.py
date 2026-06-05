"""Finance walk-forward g(x) benchmark — thesis Ch13 reproduction.

Uses ``walkforward_g_on_enriched`` on real enriched residuals when available
(``DEUP_THESIS_ENRICHED`` env or default path), otherwise a synthetic panel.

Metrics: Spearman ρ(g, rank_loss) per horizon; DEV vs FINAL date split.

Run:
    python benchmarks/run_finance_walkforward.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import DEFAULT_SEED, write_json  # noqa: E402

DEFAULT_ENRICHED = Path(
    "/Users/ursinasanderink/Downloads/AI Stock Forecast/evaluation_outputs"
    "/chapter13_v3/enriched_residuals_tabular_lgb.parquet"
)
FINAL_CUTOFF = "2024-01-01"


def _synthetic_enriched(seed: int = DEFAULT_SEED):
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for fi in range(1, 31):
        fold = f"fold_{fi:02d}"
        for _ in range(60):
            score = float(rng.normal())
            rows.append(
                {
                    "as_of_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=fi * 25),
                    "ticker": "X",
                    "stable_id": 1,
                    "horizon": 20,
                    "fold_id": fold,
                    "score": score,
                    "rank_loss": abs(float(rng.normal(scale=0.15))),
                    "vol_20d": float(rng.uniform(0.1, 0.5)),
                    "vol_60d": float(rng.uniform(0.1, 0.5)),
                    "mom_1m": float(rng.normal(scale=0.1)),
                    "adv_20d": float(rng.uniform(1e6, 5e6)),
                    "vix_percentile_252d": float(rng.uniform(0, 1)),
                    "market_regime_enc": float(rng.choice([-1.0, 0.0, 1.0])),
                    "market_vol_21d": float(rng.uniform(0.1, 0.3)),
                    "market_return_21d": float(rng.normal(scale=0.05)),
                }
            )
    return pd.DataFrame(rows)


def _run_split(enriched, *, split_name: str, min_train_folds: int = 10) -> dict[str, object]:
    from deup.domains.finance_walkforward import walkforward_g_on_enriched

    folds = sorted(
        enriched["fold_id"].unique(),
        key=lambda x: int(str(x).split("_")[1]),
    )
    min_train = min(min_train_folds, max(1, len(folds) // 3))
    preds, diag = walkforward_g_on_enriched(
        enriched,
        min_train_folds=min_train,
        horizons=[20],
        fold_sort="numeric",
    )
    if preds.empty:
        return {"split": split_name, "n_rows": 0}
    from scipy import stats

    rho = float(stats.spearmanr(preds["g_pred"], preds["rank_loss"]).statistic)
    return {
        "split": split_name,
        "n_rows": int(len(preds)),
        "n_folds": int(preds["fold_id"].nunique()),
        "spearman_rho_g_rank_loss": rho,
        "features": diag.get("features", []),
    }


def run_finance_benchmark(*, seed: int = DEFAULT_SEED, max_folds: int = 35) -> dict[str, object]:
    path = Path(os.environ.get("DEUP_THESIS_ENRICHED", DEFAULT_ENRICHED))
    synthetic = not path.exists()
    if synthetic:
        enriched = _synthetic_enriched(seed)
        data_source = "synthetic"
    else:
        import pandas as pd

        enriched = pd.read_parquet(path)
        enriched = enriched[enriched["horizon"] == 20].copy()
        folds = sorted(enriched["fold_id"].unique(), key=lambda x: int(str(x).split("_")[1]))
        keep = set(folds[-max_folds:])
        enriched = enriched[enriched["fold_id"].isin(keep)]
        data_source = f"{path} (H=20, last {max_folds} folds)"

    enriched["as_of_date"] = enriched["as_of_date"].astype("datetime64[ns]")

    # Fold-based DEV/FINAL on the selected subset (more stable than calendar cut
    # when the benchmark uses the most recent folds only).
    folds = sorted(
        enriched["fold_id"].unique(),
        key=lambda x: int(str(x).split("_")[1]),
    )
    cut = max(1, int(len(folds) * 0.7))
    dev_folds = set(folds[:cut])
    final_folds = set(folds[cut:])
    dev = enriched[enriched["fold_id"].isin(dev_folds)]
    final = enriched[enriched["fold_id"].isin(final_folds)]

    min_folds = 10 if synthetic else 20
    results = {
        "seed": seed,
        "data_source": data_source,
        "synthetic": synthetic,
        "final_cutoff": FINAL_CUTOFF,
        "thesis_reference": {
            "within_context_rho": 0.33,
            "aggregate_g_auroc_lead5d": 0.55,
            "health_index_auroc": 0.75,
            "source": "finance_cifar10c_comparison.md",
        },
        "dev": _run_split(
            dev if not dev.empty else enriched, split_name="DEV", min_train_folds=min_folds
        ),
        "final": _run_split(final, split_name="FINAL", min_train_folds=min_folds)
        if not final.empty and not synthetic
        else {"split": "FINAL", "n_rows": 0, "note": "skipped (no FINAL rows or synthetic)"},
    }
    return results


def main() -> None:
    result = run_finance_benchmark()
    write_json("finance_walkforward.json", result)
    print("=== Finance walk-forward g(x) ===")
    print(f"  data: {result['data_source']}")
    for key in ("dev", "final"):
        row = result[key]
        if row.get("n_rows", 0) > 0:
            print(
                f"  {row['split']:5s}  n={row['n_rows']}  "
                f"rho(g,rank_loss)={row.get('spearman_rho_g_rank_loss', float('nan')):.4f}"
            )


if __name__ == "__main__":
    main()
