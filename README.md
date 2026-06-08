# deup

![deup library overview](deuplibimage.png)

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

**Tabular gradient boosting** (LightGBM / XGBoost / CatBoost):

```python
from deup.domains.tabular import TabularDEUP

model = TabularDEUP(backend="lgbm", cv=5).fit(X_train, y_train)
unc = model.predict_epistemic(X_test)
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
pip install deup                 # core (numpy + scikit-learn)
pip install "deup[gbm]"          # + LightGBM (TabularDEUP backend)
pip install "deup[xgb]"          # + XGBoost
pip install "deup[catboost]"     # + CatBoost
pip install "deup[gbm-all]"      # all gradient-boosting backends
pip install "deup[finance]"      # + pandas (CrossSectionalDEUP)
pip install "deup[docs]"         # + MkDocs site locally
```

## Why this package?

The only public DEUP code was a stale research repo of notebooks — no maintained
`pip`-installable package. Major UQ libraries (`torch-uncertainty`, `uncertainty-toolbox`,
`MAPIE`) don't implement DEUP. `deup` fills that gap with **leakage-correct OOF error
construction** and **walk-forward / purged CV** for time-series and finance.

## Comparison (California housing, Spearman ρ vs realized squared error)

| Method | ρ | Notes |
|---|---|---|
| **DEUP** | **0.509** | OOF error predictor (RF base) |
| **DEUP + LightGBM** | 0.444 | `TabularDEUP(backend="lgbm")` |
| **DEUP + XGBoost** | 0.400 | `TabularDEUP(backend="xgb")` |
| **DEUP + CatBoost** | 0.407 | `TabularDEUP(backend="catboost")` |
| Ensemble disagreement | 0.460 | Bootstrap variance |
| Conformal residual | 0.447 | \|y − ŷ\| on cal split |
| Laplace (last-layer) | 0.015 | Not applicable to trees |

Full results: [Benchmarks](https://ursinasanderink.github.io/deup/benchmarks/).

## Documentation

| Topic | Link |
|---|---|
| Getting started | [docs/getting-started](https://ursinasanderink.github.io/deup/getting-started/) |
| Five-axis guide | [docs/concepts](https://ursinasanderink.github.io/deup/concepts/) |
| Domain presets | [docs/domains](https://ursinasanderink.github.io/deup/domains/) |
| Tutorials | [tabular](https://ursinasanderink.github.io/deup/tutorials/tabular-regression/) · [finance](https://ursinasanderink.github.io/deup/tutorials/finance-ranking/) · [conformal](https://ursinasanderink.github.io/deup/tutorials/classification-conformal/) · [active learning](https://ursinasanderink.github.io/deup/tutorials/active-learning/) |
| When is agg-g reliable? | [reliability](https://ursinasanderink.github.io/deup/reliability/) |
| PyTorch / TorchUncertainty | [pytorch-integration](https://ursinasanderink.github.io/deup/pytorch-integration/) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Status

**v0.4.0** — complete library: core DEUP, conformal calibration, reliability diagnostics,
domain presets (tabular GBM backends, finance, vision), benchmarks, tutorials,
TorchUncertainty integration.

## Citing

Cite this software ([`CITATION.cff`](CITATION.cff)), Lahlou *et al.* (2023) for the
DEUP method, and Sanderink (2026) for cross-sectional ranking and aggregation-
reliability extensions:

> Sanderink, U. (2026) 'When Alpha Breaks: Two-Level Uncertainty for Safe Deployment of
> Cross-Sectional Stock Rankers', *arXiv preprint* arXiv:2603.13252. Available at:
> https://arxiv.org/pdf/2603.13252

## License

Apache-2.0. See [`LICENSE`](LICENSE).


### Frozen Hugging Face LLM scenario

This forked copy includes a Scenario-A LLM extension in `deup.domains.llm`. It treats a frozen Hugging Face causal LLM as the base predictor, extracts token/logit and optional semantic-consistency features, and trains the existing `ErrorEstimator` as a DEUP predicted-risk model.

```bash
pip install -e ".[llm]"
pip install datasets tqdm
python examples/llm_scenario_a_gsm8k.py --model-id sshleifer/tiny-gpt2 --train-size 8 --test-size 4 --max-new-tokens 32
```

The resulting score is predicted task risk. It becomes a DEUP-style epistemic estimate when an aleatoric estimate is subtracted; otherwise it is the conservative proxy `u(x)=g(x)`.
