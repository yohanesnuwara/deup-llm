# deup

**Direct Epistemic Uncertainty Prediction for any scikit-learn model.**

DEUP (Lahlou et al., 2023) estimates *epistemic* uncertainty by training a secondary
**error predictor** on your model's **out-of-sample** errors. This library provides a
maintained, installable, scikit-learn-compatible implementation with first-class
support for time-series and cross-sectional workflows.

```bash
pip install deup
```

```python
from sklearn.ensemble import RandomForestRegressor
from deup import DEUPRegressor

model = DEUPRegressor(base_model=RandomForestRegressor())
model.fit(X_train, y_train)
pred, unc = model.predict(X_test, return_uncertainty=True)

# Active learning: pick the 10 most uncertain points
idx = model.acquire(X_pool, k=10)
```

For classification or cross-sectional ranking:

```python
from deup import DEUPClassifier, DEUPRanker

clf = DEUPClassifier(base_model=my_classifier).fit(X, y)
ranker = DEUPRanker(base_model=my_ranker).fit(X, y, groups=dates)
```

## Why deup?

- **Works with models you already use** — RandomForest, LightGBM, linear models, etc.
- **Leakage-correct by default** — out-of-fold errors, not in-sample residuals
- **Time-series ready** — `PurgedWalkForward` with embargo for walk-forward panels
- **Benchmarked** — see [Benchmarks](benchmarks.md)

## Attribution

DEUP the *method* is due to Lahlou, Jain, Nekoei, Butoi, Bertin, Rector-Brooks,
Korablyov, and Bengio (2023, TMLR). This package is an independent library
implementation; please cite both the paper and this software (`CITATION.cff`).

## Next steps

- [Getting started](getting-started.md) — install, fit, interpret uncertainty
- [Theory & math](theory.md) — risk decomposition, DEUP algorithms, stationarizing features
- [Losses & transforms](losses.md) — squared, Brier, pinball, rank, log/asinh targets
- [Feature builders](features.md) — density, variance, seen-bit for $g(x)$
