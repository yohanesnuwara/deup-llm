# Tutorial: Classification + conformal intervals

**Goal:** classify with DEUP epistemic scores, then wrap them in calibrated
prediction sets with finite-sample coverage.

## Fit classifier

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deup import DEUPClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
X_tr, X_cal, y_tr, y_cal = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)

clf = DEUPClassifier(
    base_model=RandomForestClassifier(n_estimators=80, random_state=0),
    cv=5,
    random_state=0,
)
clf.fit(X_tr, y_tr)
```

## Epistemic uncertainty

```python
pred, unc = clf.predict(X_te, return_uncertainty=True)
proba = clf.predict_proba(X_te)
```

`unc` comes from `g` trained on out-of-fold log-loss (or Brier if configured).

## Conformal calibration (separate cal split)

```python
clf.calibrate(X_cal, y_cal, method="normalized", alpha=0.1)
result = clf.predict_interval(X_te)
result.lower, result.upper
```

- `method="normalized"` scales intervals by `g(x)` — narrow where confident
- `method="mondrian"` — per-group intervals (pass `groups=` if needed)
- Target coverage: **1 − α** (marginal, distribution-free)

See [Conformal calibration](../calibration.md) for math and MAPIE interop.

## Next

- [Active learning](active-learning.md)
- [Theory: Algorithm 2](../theory.md)
