# deup

**Direct Epistemic Uncertainty Prediction for any scikit-learn model.**

DEUP (Lahlou et al., 2023) estimates *epistemic* uncertainty by training a secondary
**error predictor** on your model's **out-of-sample** errors. This library provides a
maintained, installable, scikit-learn-compatible implementation with first-class
support for time-series, cross-sectional ranking, conformal intervals, and
research-grade reliability diagnostics.

```bash
pip install deup
pip install "deup[finance,gbm]"   # pandas + LightGBM presets
```

```python
from sklearn.ensemble import RandomForestRegressor
from deup import DEUPRegressor

model = DEUPRegressor(base_model=RandomForestRegressor())
model.fit(X_train, y_train)
pred, unc = model.predict(X_test, return_uncertainty=True)
```

Cross-sectional finance panel:

```python
from deup.domains.finance import CrossSectionalDEUP

model = CrossSectionalDEUP(horizon=20).fit(panel_df)
pred, unc = model.predict(test_df, return_uncertainty=True)
health = model.health_report(test_df)
```

## Why deup?

- **Works with models you already use** — sklearn, LightGBM, any `fit`/`predict` API
- **Leakage-correct by default** — out-of-fold errors (Algorithm 2), purged walk-forward
- **Time-series & ranking** — `DEUPRanker`, rank-geometry residualization, `HealthIndex`
- **Calibrated intervals** — split-conformal `predict_interval()` + MAPIE interop
- **Benchmarked** — DEUP beats ensembles/conformal on tabular; N-sweep validates Finding 1

## Documentation map

| Topic | Page |
|---|---|
| Quickstart | [Getting started](getting-started.md) |
| Math & algorithms | [Theory & math](theory.md) |
| When is agg-g reliable? | [Aggregation reliability](reliability.md) |
| Finance / vision presets | [Domain presets](domains.md) |
| Thesis migration | [Migrating from thesis](migration.md) |
| Benchmarks & N-sweep | [Benchmarks](benchmarks.md) |

## Attribution

DEUP the *method* is due to Lahlou, Jain, Nekoei, Butoi, Bertin, Rector-Brooks,
Korablyov, and Bengio (2023, TMLR). Ranking adaptations and two-level deployment
build on Sanderink (2026). Please cite both papers and this software (`CITATION.cff`).

## Status

**v0.3** — estimators, features, conformal calibration, diagnostics (Findings 1–2),
domain presets, thesis parity, benchmark suite. See [CHANGELOG](https://github.com/ursinasanderink/deup/blob/main/CHANGELOG.md).
