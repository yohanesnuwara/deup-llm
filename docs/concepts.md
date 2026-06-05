# The five axes — configuring DEUP for any use case

DEUP is a **meta-algorithm**: the same orchestration wraps any base model. Your use
case differs only along five pluggable axes (see `ARCHITECTURE.md` in the repo).

| # | Axis | What you choose | Examples |
|---|---|---|---|
| 1 | **Task** | Estimator class | regression, classification, ranking |
| 2 | **Loss / error target** | What `g` predicts | squared, log-loss, rank, pinball |
| 3 | **Grouping** | Row structure | i.i.d., panel-by-date, by-entity |
| 4 | **Out-of-sample scheme** | How OOF errors are collected | `KFold`, `PurgedWalkForward` |
| 5 | **g-features** | What `g` sees | raw X, density, variance, panel cols |

Pick one row from the map below — everything else is wiring.

## Use-case map

| Use case | Task | Loss | Group | CV | g-features |
|---|---|---|---|---|---|
| Generic tabular | `TabularDEUP` | squared | i.i.d. | `KFold` | raw X + density |
| Tabular + LightGBM/XGB/CatBoost | `TabularDEUP(backend=…)` | squared | i.i.d. | `KFold` | raw X + density |
| Cross-sectional ranker | `DEUPRanker` | rank | by-date | `PurgedWalkForward` | score, vol, regime |
| Time-series forecast | `DEUPRegressor` | squared | time | walk-forward | residual, vol |
| Classification + intervals | `DEUPClassifier` | log-loss | i.i.d. | `KFold` | raw X |
| Vision / OOD batches | `VisionDEUP` | log-loss | i.i.d. | `KFold` | embedding density |
| Active learning | any | predicted error | i.i.d. | `KFold` | density, distance |

## Presets vs primitives

**Presets** (fastest path):

```python
from deup.domains.tabular import TabularDEUP
from deup.domains.finance import CrossSectionalDEUP
from deup.domains.vision import VisionDEUP
```

**Primitives** (full control):

```python
from deup.core import OOFErrorCollector, ErrorEstimator, FeaturePipeline
from deup import DEUPRegressor
```

## Aggregation guard (Finding 1)

Individual-level `g(x)` is reliable across domains. **Context-level** `mean(g)` is only
trustworthy at high N with exchangeable errors. See
[When is aggregated DEUP reliable?](reliability.md) and the
[N-sweep benchmark](benchmarks.md).

For low-N / non-i.i.d. (finance, time-series), use
[`HealthIndex`](api/diagnostics.md) instead of raw `mean(g)`.

## Further reading

- [Theory & math](theory.md) — algorithms and risk decomposition
- [Tutorials](tutorials/tabular-regression.md) — hands-on walkthroughs
- [Domain presets](domains.md) — finance, tabular, vision
