# Benchmarks

Reproducible uncertainty-quality comparisons for `deup`. All scripts use **seed=42**
and write JSON tables under `benchmarks/results/`.

## Quick run

```bash
pip install -e ".[dev,benchmark,gbm,finance]" pyarrow
python benchmarks/run_all.py
```

Individual scripts:

| Script | Output |
|---|---|
| `run_regression_benchmark.py` | `results/regression_benchmark.json` |
| `run_n_sweep.py` | `results/n_sweep.json`, `results/n_sweep.png` |
| `run_cifar_proxy.py` | `results/cifar_proxy.json` |
| `run_finance_walkforward.py` | `results/finance_walkforward.json` |

Set `DEUP_FINANCE_ENRICHED=/path/to/enriched_residuals.parquet` for a real finance panel;
otherwise a synthetic panel is used.

---

## 1. Tabular regression — DEUP vs baselines (California housing)

**Question:** which method best *ranks* test points by realized squared error?

**Metric:** Spearman ρ(uncertainty, (y−ŷ)²) on held-out test (n=4,128). **Higher is better.**

| Method | Spearman ρ | Notes |
|---|---:|---|
| **DEUP** | **0.509** | `DEUPRegressor` + RF base, 5-fold OOF |
| **DEUP + LightGBM** | 0.444 | `TabularDEUP(backend="lgbm")` |
| **DEUP + XGBoost** | 0.400 | `TabularDEUP(backend="xgb")` |
| **DEUP + CatBoost** | 0.407 | `TabularDEUP(backend="catboost")` |
| Ensemble disagreement | 0.460 | 5 bootstrap RF members, prediction variance |
| Conformal residual | 0.447 | Cal-set model for \|residual\| magnitude |
| Laplace (BayesianRidge) | 0.015 | Gaussian posterior variance (poor on this task) |

DEUP wins on this tabular regression benchmark — the epistemic score tracks which
predictions are likely wrong better than ensemble or conformal-residual baselines.

*Split: DEV (80/20 train/test). Last run: seed=42.*

---

## 2. N-sweep — aggregation reliability (headline)

**Question:** how does AUROC(**mean g** per context, bad context) scale with N and
dependence structure?

![N-sweep](../docs/assets/n_sweep.png)

### i.i.d. synthetic contexts (exchangeable)

| N / context | # contexts | AUROC(agg_g) | ρ(agg_g, agg error) |
|---:|---:|---:|---:|
| 50 | 800 | 0.810 | 0.578 |
| 200 | 200 | 0.916 | 0.810 |
| 1,000 | 40 | **0.960** | **0.927** |
| 5,000 | 30 | 0.898 | 0.899 |
| 10,000 | 30 | 0.920 | 0.914 |

At high N with i.i.d. errors, aggregated g detects bad contexts at AUROC ≈ **0.92–0.96**,
consistent with CIFAR-10-C batch literature (~**0.955** at N≈10,000).

### Low-N / autocorrelated (finance-like)

| Regime | N | AUROC(agg_g) | AUROC(HealthIndex) |
|---|---:|---:|---:|
| Autocorr + small N | 50 | **0.435** | **1.000** |

Raw `mean(g)` is near-chance (~**0.55** reference on cross-sectional finance); the
composite **HealthIndex** recovers context detection (≈ **0.75** AUROC on a published
finance holdout).

See [When is aggregated DEUP reliable?](https://ursinasanderink.github.io/deup/reliability/).

---

## 3. CIFAR-10-C aggregation proxy

Full CIFAR-10-C training requires GPU + torch. The proxy simulates **high-N i.i.d. batches**
(scaled to 800×30 here for CPU; literature uses N≈10k images/batch).

| Metric | Proxy | Literature reference |
|---|---:|---:|
| AUROC(agg_g, broken batch) — oracle g | **1.000** | 0.955 |
| AUROC(agg_g, broken batch) — VisionDEUP | smoke | 0.955 |
| ρ(agg_g, batch error) | — | 0.926 |

Oracle g validates the aggregation metric at high N; `VisionDEUP` exercises the
embedding→density→variance→g pipeline on synthetic tensors.

---

## 4. Finance walk-forward g(x)

Walk-forward error predictor on a cross-sectional ranker panel via
`walkforward_g_on_enriched` (H=20).

| Split | n predictions | ρ(g, rank_loss) | Reference |
|---|---:|---:|---|
| DEV (70% folds) | 33,887 | 0.249 | within-context ρ ≈ 0.33 |
| FINAL (30% folds) | 15,969 | 0.169 | agg-g AUROC ≈ 0.55 |

---

## Comparison summary

| Task | Best method | Key metric |
|---|---|---|
| Tabular uncertainty ranking | **DEUP** | Spearman 0.509 |
| High-N i.i.d. context detection | **agg g** | AUROC ≈ 0.96 |
| Low-N finance-like context | **HealthIndex** | AUROC 1.0 (proxy) / 0.75 (ref) |
| CIFAR batch OOD (literature) | **agg g** | AUROC 0.955 |

---

## Not yet benchmarked

- MC-Dropout (requires `[torch]`)
- Full GPU CIFAR-10-C end-to-end (proxy only in CI)
- MAPIE interval efficiency benchmark

## Future model presets (planned)

- **XGBoost / CatBoost** tabular presets
- **torchvision** ResNet-18 embeddings → `VisionDEUP`
- **HuggingFace** encoder preset (text/sentence → DEUP)
- **PyTorch Lightning** training-loop integration
