# Benchmarks

Reproducible uncertainty-quality comparisons for `deup`.

## Quick run

```bash
pip install -e ".[dev]"
python benchmarks/run_regression_benchmark.py
```

Results are written to `benchmarks/results/regression_benchmark.json`.

## Regression benchmark (California housing)

**Question:** which method best *ranks* test points by realized squared error?

**Metric:** Spearman correlation between each method's uncertainty score and
`(y - ŷ)²` on a held-out test set (n=4,128). Higher is better.

| Method | Spearman | Notes |
|---|---:|---|
| **DEUP** | **0.510** | `DEUPRegressor` + RF base |
| Ensemble disagreement | 0.460 | 5 bootstrap RF members, prediction variance |
| Conformal residual | 0.447 | Cal-set model for `\|residual\|` magnitude |

*Last run: local dev checkout, seed=0, commit `P-min-bench`.*

DEUP wins on this tabular regression task — the uncertainty score tracks which
predictions are likely to be wrong better than the two sklearn-only baselines.

### N-sweep teaser (context-level aggregation)

Synthetic heteroscedastic panels; for each context size N we report Spearman
between **mean g(x)** per context and **mean realized squared error** per context.

| N / context | # contexts | agg Spearman |
|---:|---:|---:|
| 10 | 800 | 0.611 |
| 50 | 160 | 0.577 |
| 200 | 40 | 0.664 |
| 1000 | 20 | 0.498 |

This is a **teaser**, not the full finance/CIFAR cross-domain study from the thesis.
At very small numbers of contexts (N=1000 → only 20 contexts) the aggregate
estimate is noisy. The full `AggregationReliability` diagnostic (v0.2) will formalize
when aggregated DEUP is trustworthy.

## Not yet benchmarked (v0.2+)

- MC-Dropout (requires `[torch]`)
- MAPIE interop
- Time-series / purged walk-forward on real finance panel
- CIFAR-10-C OOD reproduction
