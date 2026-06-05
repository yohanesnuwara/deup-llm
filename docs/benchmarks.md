# Benchmarks

Reproducible uncertainty-quality comparisons for `deup`. All scripts use **seed=42**
and write JSON tables under `benchmarks/results/`.

## Quick run

```bash
pip install -e ".[dev,benchmark,gbm,finance]" pyarrow
python benchmarks/run_all.py
```

See the full write-up in [BENCHMARKS.md](https://github.com/ursinasanderink/deup/blob/main/BENCHMARKS.md)
in the repository root (tables are committed from the last benchmark run).

---

## Tabular regression (California housing)

| Method | Spearman ρ | Notes |
|---|---:|---|
| **DEUP** | **0.509** | `DEUPRegressor` + RF |
| Ensemble disagreement | 0.460 | Bootstrap variance |
| Conformal residual | 0.447 | Cal-set \|residual\| model |
| Laplace (BayesianRidge) | 0.015 | Posterior variance |

---

## N-sweep — Finding 1 (headline)

![N-sweep AUROC vs context size](assets/n_sweep.png)

**i.i.d. contexts:** AUROC(agg_g) rises to **≈0.96** at N≈1,000–10,000 (CIFAR thesis ref ≈0.955).

**Low-N autocorrelated:** AUROC(agg_g)≈**0.43** (finance ref ≈0.55); **HealthIndex** recovers
to AUROC≈**1.0** on the synthetic proxy (thesis ref ≈0.75 on real FINAL holdout).

Details: [Aggregation reliability](reliability.md).

---

## CIFAR & finance

- **CIFAR proxy:** oracle agg-g AUROC **1.0** on high-N i.i.d. batch simulation (thesis **0.955**)
- **Finance walk-forward:** ρ(g, rank_loss)=**0.25** DEV / **0.17** FINAL on Ch13 v3 subset;
  exact parity with frozen thesis `g_pred` documented in [Migrating from thesis](migration.md)

---

## Future work

XGBoost/CatBoost tabular presets; torchvision ResNet-18 → `VisionDEUP`; HuggingFace encoders;
PyTorch Lightning hooks — see the project roadmap.
