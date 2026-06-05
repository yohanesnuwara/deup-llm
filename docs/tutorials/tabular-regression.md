# Tutorial: Tabular regression

**Goal:** wrap an existing sklearn regressor with DEUP and rank test points by
expected error.

## Setup

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deup import DEUPRegressor

X, y = fetch_california_housing(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
```

## Fit DEUP

```python
model = DEUPRegressor(
    base_model=RandomForestRegressor(n_estimators=80, random_state=0),
    cv=5,
    random_state=0,
)
model.fit(X_tr, y_tr)
```

Under the hood: `OOFErrorCollector` gathers out-of-fold squared errors, then
`ErrorEstimator` trains `g` to predict them.

## Predict with uncertainty

```python
pred, unc = model.predict(X_te, return_uncertainty=True)
```

- `pred` — base model point prediction (refit on all training data)
- `unc` — epistemic estimate `g(x) ≥ 0` (higher = less trustworthy)

## Optional: tabular preset with density features

```python
from deup.domains.tabular import TabularDEUP

preset = TabularDEUP(cv=5, random_state=0)
preset.fit(X_tr, y_tr)
unc = preset.predict_epistemic(X_te)
```

## Benchmark context

On California housing, DEUP beats ensemble disagreement and conformal-residual
baselines for ranking realized squared error (Spearman ρ ≈ **0.51**). See
[Benchmarks](../benchmarks.md).

## Next

- [Classification + conformal intervals](classification-conformal.md)
- [Active learning](active-learning.md)
