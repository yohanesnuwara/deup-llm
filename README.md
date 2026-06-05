# deup

**Direct Epistemic Uncertainty Prediction (DEUP) for any scikit-learn model — with first-class, leakage-correct time-series support.**

[![PyPI](https://img.shields.io/pypi/v/deup)](https://pypi.org/project/deup/)
[![CI](https://github.com/ursinasanderink/deup/actions/workflows/ci.yml/badge.svg)](https://github.com/ursinasanderink/deup/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://ursinasanderink.github.io/deup/)

DEUP estimates *epistemic* uncertainty by training a secondary **error predictor** on
your model's **out-of-sample** errors — no ensembles, no Bayesian retraining, works
with the model you already use. The method is due to
[Lahlou et al., 2023 (TMLR)](https://openreview.net/forum?id=eGLdVRvvfQ); this package
is a maintained, installable, scikit-learn-compatible implementation of it.

Repository: <https://github.com/ursinasanderink/deup> · Docs: <https://ursinasanderink.github.io/deup/>

## Quickstart

```python
from sklearn.ensemble import RandomForestRegressor
from deup import DEUPRegressor

model = DEUPRegressor(base_model=RandomForestRegressor())
model.fit(X_train, y_train)

pred, unc = model.predict(X_test, return_uncertainty=True)
```

For time-series / cross-sectional data, pass a leakage-safe splitter:

```python
from deup.splitters import PurgedWalkForward

model = DEUPRegressor(base_model=my_model, cv=PurgedWalkForward(embargo=5))
```

## Install

```bash
pip install deup            # core (numpy + scikit-learn)
pip install "deup[gbm]"     # + LightGBM error predictor
pip install "deup[docs]"    # + MkDocs site locally
```

## Why this exists

The only public DEUP code is a 3-year-old research repo of notebooks; no maintained
`pip`-installable package existed until now. Major UQ libraries
(`torch-uncertainty`, `uncertainty-toolbox`, `MAPIE`) don't implement DEUP. `deup`
fills that gap with **correct out-of-fold error construction** for time-series /
cross-sectional data.

On California housing (v0.1 benchmark), DEUP uncertainty ranks test errors better
than ensemble disagreement or a conformal residual baseline — see [BENCHMARKS.md](BENCHMARKS.md).

## Status / roadmap

**v0.2 (current):** `DEUPRegressor`, `DEUPClassifier`, `DEUPRanker`, `acquire`,
OOF collector, splitters, full loss registry, feature builders, `ErrorEstimator`,
aleatoric decomposition, rank residualization, conformal intervals
(`predict_interval` + MAPIE interop), benchmark, docs.

**Next:** aggregation-reliability diagnostics (P9).

## Citing

If you use `deup`, please cite both this software (see [`CITATION.cff`](CITATION.cff))
and the original DEUP paper (Lahlou et al., 2023).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
