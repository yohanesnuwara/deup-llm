# deup

**Direct Epistemic Uncertainty Prediction (DEUP) for any scikit-learn model — with first-class, leakage-correct time-series support.**

[![PyPI](https://img.shields.io/pypi/v/deup)](https://pypi.org/project/deup/)
[![CI](https://github.com/ursinasanderink/deup/actions/workflows/ci.yml/badge.svg)](https://github.com/ursinasanderink/deup/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://ursinasanderink.github.io/deup/)

DEUP estimates *epistemic* uncertainty by training a secondary **error predictor** on
your model's **out-of-sample** errors — no ensembles, no Bayesian retraining, works
with the model you already use.

**Method credit:** DEUP is due to
[Lahlou et al., 2023 (TMLR)](https://openreview.net/forum?id=eGLdVRvvfQ). This package
is a maintained, benchmarked, scikit-learn-compatible implementation with time-series /
cross-sectional finance support and aggregation-reliability diagnostics.

Repository: <https://github.com/ursinasanderink/deup> · Docs: <https://ursinasanderink.github.io/deup/>

## Quickstart

```python
from sklearn.ensemble import RandomForestRegressor
from deup import DEUPRegressor

model = DEUPRegressor(base_model=RandomForestRegressor())
model.fit(X_train, y_train)

pred, unc = model.predict(X_test, return_uncertainty=True)
```

**Time-series / cross-sectional finance** (flagship preset):

```python
from deup.domains.finance import CrossSectionalDEUP

model = CrossSectionalDEUP(horizon=20).fit(panel_df)
pred, unc = model.predict(test_df, return_uncertainty=True)
health = model.health_report(test_df)
```

## Install

```bash
pip install deup            # core (numpy + scikit-learn)
pip install "deup[gbm]"     # + LightGBM error predictor
pip install "deup[finance]" # + pandas (CrossSectionalDEUP)
pip install "deup[docs]"    # + MkDocs site locally
```

## Why this package?

The only public DEUP code was a stale research repo of notebooks — no maintained
`pip`-installable package. Major UQ libraries (`torch-uncertainty`, `uncertainty-toolbox`,
`MAPIE`) don't implement DEUP. `deup` fills that gap with **leakage-correct OOF error
construction** and **walk-forward / purged CV** for time-series and finance.

## Comparison (California housing, Spearman ρ vs realized squared error)

| Method | ρ | Notes |
|---|---|---|
| **DEUP** | **0.509** | OOF error predictor |
| Ensemble disagreement | 0.460 | Bootstrap variance |
| Conformal residual | 0.447 | \|y − ŷ\| on cal split |
| Laplace (last-layer) | 0.015 | Not applicable to trees |

Full results: [Benchmarks](https://ursinasanderink.github.io/deup/benchmarks/).

## Documentation

| Topic | Link |
|---|---|
| Getting started | [docs/getting-started](https://ursinasanderink.github.io/deup/getting-started/) |
| Five-axis guide | [docs/concepts](https://ursinasanderink.github.io/deup/concepts/) |
| Tutorials | [tabular](https://ursinasanderink.github.io/deup/tutorials/tabular-regression/) · [finance](https://ursinasanderink.github.io/deup/tutorials/finance-ranking/) · [conformal](https://ursinasanderink.github.io/deup/tutorials/classification-conformal/) · [active learning](https://ursinasanderink.github.io/deup/tutorials/active-learning/) |
| When is agg-g reliable? | [reliability](https://ursinasanderink.github.io/deup/reliability/) |
| PyTorch / TorchUncertainty | [pytorch-integration](https://ursinasanderink.github.io/deup/pytorch-integration/) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Status

**v0.3.2** — full stack through P15 TorchUncertainty integration.

## Citing

Cite this software ([`CITATION.cff`](CITATION.cff)) **and** Lahlou et al. (2023).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
