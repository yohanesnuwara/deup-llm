# Getting started

## Install

```bash
pip install deup
```

Optional extras:

```bash
pip install "deup[gbm]"       # LightGBM tabular backend
pip install "deup[xgb]"       # XGBoost tabular backend
pip install "deup[catboost]"  # CatBoost tabular backend
pip install "deup[gbm-all]"   # all gradient-boosting backends
pip install "deup[finance]"   # CrossSectionalDEUP (pandas)
pip install "deup[docs]"      # MkDocs site locally
```

## Quickstart — tabular regression

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deup import DEUPRegressor

X, y = fetch_california_housing(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

model = DEUPRegressor(
    base_model=RandomForestRegressor(n_estimators=100, random_state=0),
    cv=5,
    random_state=0,
)
model.fit(X_tr, y_tr)

pred, unc = model.predict(X_te, return_uncertainty=True)
```

- `pred` — point prediction from the base model (refit on all training data)
- `unc` — estimated epistemic uncertainty `g(x)` (predicted out-of-sample error)

Higher `unc` means the model is likely to be wrong at that input. On California
housing (v0.1 benchmark), Spearman(`unc`, realized squared error) ≈ **0.51**, beating
ensemble disagreement (0.46) and a conformal residual baseline (0.45).

## Time-series / walk-forward

For ordered or panel data, pass a leakage-safe splitter:

```python
from deup import DEUPRegressor
from deup.splitters import PurgedWalkForward

model = DEUPRegressor(
    base_model=my_model,
    cv=PurgedWalkForward(n_splits=5, embargo=5, min_train_size=20),
)
model.fit(X, y, groups=dates)  # dates = cross-section label per row
pred, unc = model.predict(X_test, return_uncertainty=True)
```

`PurgedWalkForward` keeps each date's full cross-section together and drops an
**embargo** between train and test to prevent look-ahead leakage.

## Cross-sectional ranking

Use :class:`~deup.estimators.DEUPRanker` — rank loss, walk-forward CV, and rank-geometry
residualization ON by default (Finding 3):

```python
from deup import DEUPRanker

model = DEUPRanker(base_model=my_ranker, cv=5)
model.fit(X, y, groups=dates)
pred, unc = model.predict(X_test, return_uncertainty=True, groups=test_dates)
```

Set `residualize_rank=False` to disable decoupling (not recommended for rankers).

## Classification

```python
from deup import DEUPClassifier
from sklearn.ensemble import RandomForestClassifier

model = DEUPClassifier(base_model=RandomForestClassifier(), cv=5)
model.fit(X, y)
pred, unc = model.predict(X_test, return_uncertainty=True)
proba = model.predict_proba(X_test)
```

Default loss is `logloss`; use `loss="brier"` for Brier score targets.

## Active learning — `acquire`

Select the `k` pool points with highest epistemic uncertainty (DEUP paper, Sec. 3.2):

```python
idx = model.acquire(X_pool, k=10)
# or with uncertainty values:
idx, unc = model.acquire(X_pool, k=10, return_uncertainty=True)
```

For rankers pass `groups=` so residualization is correct on panel data.

## Aleatoric decomposition

Subtract an aleatoric floor for a cleaner epistemic signal
$\hat{e}(x) = \max(0, g(x) - a(x))$:

```python
from deup.core import Heteroscedastic

model = DEUPRegressor(
    base_model=my_model,
    aleatoric=Heteroscedastic(k=20),
)
model.fit(X, y)
unc = model.predict_epistemic(X)  # max(0, g - a)
```

See [Decomposition](decomposition.md) for details.

## Prediction intervals (conformal)

Wrap the epistemic score in calibrated intervals with marginal coverage $1-\alpha$:

```python
model = DEUPRegressor(base_model=my_model).fit(X_train, y_train)
model.calibrate(X_cal, y_cal, alpha=0.1)   # held-out split!
interval = model.predict_interval(X_test)
interval.lower, interval.upper
```

See [Conformal calibration](calibration.md) for methods (`normalized`, `mondrian`,
`cqr`) and MAPIE interop.

## Aggregating `g` over contexts (read this first)

Averaging `g` into a context-level signal (`mean(g)` per day/batch/group) is only
trustworthy at high N with i.i.d. errors. Before relying on it, check:

```python
from deup.diagnostics import should_trust_aggregate

verdict = should_trust_aggregate(g, groups)
print(verdict.trustworthy, verdict.reason)
```

For low-N / non-i.i.d. (time-series, finance), use the composite `HealthIndex`
instead. See [When is aggregated DEUP reliable?](reliability.md).

## Cross-sectional finance preset

For panel data (one row per date × asset), use the flagship preset:

```python
from deup.domains.finance import CrossSectionalDEUP

model = CrossSectionalDEUP(horizon=20, cv=5, embargo=1).fit(panel_df)
model.calibrate(cal_df, alpha=0.1)
pred, unc = model.predict(test_df, return_uncertainty=True)
health = model.health_report(test_df)
```

See [Domain presets](domains.md) for tabular and vision presets too.

## Target stabilization

Heavy-tailed error targets are stabilized before training `g`:

```python
# default: log(error + eps)
model = DEUPRegressor(target_transform="log")

# robust alternative for very heavy tails
model = DEUPRegressor(target_transform="asinh")

# raw errors (no transform)
model = DEUPRegressor(target_transform="none")
```

## How it works (and the refit assumption)

DEUP fits two models:

1. **Base model `f`** — predicts `y` from `x` (your existing model).
2. **Error predictor `g`** — predicts how wrong `f` is at `x`.

The subtle part is generating honest training targets for `g`. If you trained `g` on
the residuals of an `f` that had already *seen* those rows, the residuals would be
optimistically small and `g` would systematically **under**-estimate uncertainty —
the canonical DEUP failure mode (Lahlou et al., 2023, Sec. 3.2). `deup` avoids this
with leakage-correct out-of-fold collection (`OOFErrorCollector`, the paper's
Algorithm 2): for each CV fold a fresh clone of `f` is fit on the train rows and used
to predict the *held-out* rows, so every error target is genuinely out-of-sample.

By default the collector then refits `f` on **all** data for deployment. So `g` is
trained on the errors of fold models `f₋ₖ` (each fit on a strict subset) but paired at
inference with the full-data `f`. This is the standard DEUP / stacking assumption:
`g` describes the error of a *slightly smaller* model. For reasonable fold counts the
gap is small; under walk-forward the fold models are legitimately smaller (expanding
window), which is the realistic operating regime. Pass `refit_on_all=False` (or a
pre-fit estimator) if you want to disable the final refit.

## What v0.3 includes

**Included:** estimators, conformal calibration, aggregation-reliability diagnostics
(`AggregationReliability`, `HealthIndex`), domain presets (`CrossSectionalDEUP`,
`TabularDEUP`, `VisionDEUP`), benchmark suite, and
[step-by-step tutorials](tutorials/tabular-regression.md).

Configure any use case via the [five-axis guide](concepts.md).

## Run the benchmark locally

```bash
git clone https://github.com/ursinasanderink/deup.git
cd deup
pip install -e ".[dev]"
python benchmarks/run_regression_benchmark.py
```

Results land in `benchmarks/results/regression_benchmark.json`.
