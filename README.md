# deup

**Direct Epistemic Uncertainty Prediction (DEUP) for any scikit-learn model — with first-class, leakage-correct time-series support.**

> ⚠️ **Pre-release (v0.0.1.dev).** The API below is the v0.1 target. This repo is
> currently scaffolding; follow along as the v0.1 critical path lands.

DEUP estimates *epistemic* uncertainty by training a secondary **error predictor** on
your model's **out-of-sample** errors — no ensembles, no Bayesian retraining, works
with the model you already use. The method is due to
[Lahlou et al., 2023 (TMLR)](https://openreview.net/forum?id=eGLdVRvvfQ); this package
is a maintained, installable, scikit-learn-compatible implementation of it.

Repository: <https://github.com/ursinasanderink/deup>

## Why this exists

The only public DEUP code is a 3-year-old research repo of notebooks; no maintained
`pip`-installable package exists, and the major UQ libraries
(`torch-uncertainty`, `uncertainty-toolbox`, `MAPIE`) don't implement DEUP. `deup`
fills that gap, and adds the thing those libraries don't have: **correct out-of-fold
error construction for time-series / cross-sectional data**, where naive
implementations silently leak and overstate accuracy.

## Quickstart (v0.1 target API)

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
pip install "deup[torch]"   # + neural error predictor / GP variance features
```

## Status / roadmap

v0.1 (in progress): leakage-correct out-of-fold errors (KFold / TimeSeriesSplit /
PurgedWalkForward), `DEUPRegressor` with the ergonomic API above, one benchmark
notebook (DEUP vs. conformal / ensembles / MC-dropout), docs site.

Later: classification & ranking, conformal-calibrated intervals, density/GP features,
aggregation-reliability diagnostics, domain presets (finance / vision).

## Citing

If you use `deup`, please cite both this software (see [`CITATION.cff`](CITATION.cff))
and the original DEUP paper (Lahlou et al., 2023).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
