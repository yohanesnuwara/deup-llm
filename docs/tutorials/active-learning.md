# Tutorial: Active learning with `acquire`

**Goal:** select the most informative unlabeled points to label next, using DEUP's
epistemic score (Lahlou et al., Sec. 3.2).

## Train on a labeled pool

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deup import DEUPClassifier

X, y = make_classification(n_samples=800, n_features=12, random_state=0)
X_lab, X_pool, y_lab, y_pool = train_test_split(X, y, test_size=0.6, random_state=0)

model = DEUPClassifier(
    base_model=RandomForestClassifier(n_estimators=60, random_state=0),
    cv=5,
    random_state=0,
)
model.fit(X_lab, y_lab)
```

## Acquire top-k uncertain points

```python
k = 10
idx = model.acquire(X_pool, k=k)
# idx — indices into X_pool with highest epistemic uncertainty

idx, unc = model.acquire(X_pool, k=k, return_uncertainty=True)
```

This is the DEUP paper's active-learning hook: label `X_pool[idx]` next.

## Ranking panels

For `DEUPRanker`, pass `groups=` so rank residualization is correct:

```python
from deup import DEUPRanker

ranker = DEUPRanker().fit(X_lab, y_lab, groups=dates_lab)
idx = ranker.acquire(X_pool, k=5, groups=dates_pool)
```

## Next

- [Tabular regression](tabular-regression.md)
- [Concepts](../concepts.md)
