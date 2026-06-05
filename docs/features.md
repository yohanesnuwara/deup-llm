# Feature builders for $g(x)$

The error predictor $g$ in DEUP can use **stationarizing features**
$\phi_{z^N}(x)$ beyond raw inputs (Lahlou *et al.*, 2023, Sec. 3.2). Each builder
is a scikit-learn `TransformerMixin` that **fits on training data only** — the same
leakage discipline as `OOFErrorCollector` (Finding 4).

See [Theory](theory.md) for the mathematical definitions.

## Quick example

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from deup.core.features import (
    DensityFeature,
    DistanceToTrain,
    FeaturePipeline,
    RawFeatures,
    SeenBit,
)

pipe = FeaturePipeline([
    ("raw", RawFeatures()),
    ("density", DensityFeature(method="mahalanobis")),
    ("dist", DistanceToTrain(k=5)),
    ("seen", SeenBit(atol=1e-8)),
])

X_train = np.random.default_rng(0).normal(size=(500, 8))
X_test = np.random.default_rng(1).normal(size=(50, 8))

phi_train = pipe.fit_transform(X_train)
phi_test = pipe.transform(X_test)
print(phi_train.shape, phi_test.shape)  # (500, 8+1+1+1), (50, ...)
```

## Builders

| Class | Output | Methods / notes |
|---|---|---|
| `RawFeatures` | $x$ | passthrough |
| `DensityFeature` | $\log \hat{q}(x)$ column | `mahalanobis`, `knn`, `kde`; `flow` requires `[torch]` |
| `VarianceFeature` | $\log \hat{V}(x)$ column | `ensemble` (bootstrap); `gp` requires `[torch]` |
| `DistanceToTrain` | $k$-th NN distance | default `k=5` |
| `SeenBit` | $s \in \{0,1\}$ | exact / `atol` duplicate detection |
| `ResidualMagnitude` | kNN-smoothed $\|y-f(x)\|$ | needs `estimator` + `y` at `fit` |

### DensityFeature

```python
# Diagonal Gaussian — matches thesis GaussianDensity.log_prob (Lee et al. 2018)
DensityFeature(method="mahalanobis")

# k-NN distance proxy: log q ≈ -log(d_k + ε)
DensityFeature(method="knn", k=5)

# sklearn KernelDensity
DensityFeature(method="kde", bandwidth=1.0)
```

!!! warning "Finding 3"
    Density can be **informative null** in homogeneous tabular panels. Ablate with
    `FeaturePipeline` column importances or drop if $\Delta\rho < 0.005$.

### VarianceFeature (ensemble)

Fits `n_estimators` bootstrap replicas of a base model and returns
$\log(\mathrm{Var}_j f_j(x) + \varepsilon)$.

```python
VarianceFeature(
    method="ensemble",
    estimator=RandomForestRegressor(n_estimators=50, random_state=0),
    n_estimators=10,
)
```

### ResidualMagnitude

At `fit(X, y)` stores training residuals $|y - f(x)|$. At `transform(X)` returns
the mean residual magnitude among $k$ nearest training neighbors — a local error prior
when $y$ is unavailable at inference.

```python
ResidualMagnitude(
    estimator=RandomForestRegressor(),
    k=5,
).fit(X_train, y_train)
```

## FeaturePipeline

`FeaturePipeline` horizontally stacks named builders (FeatureUnion-style). Names appear
in `get_feature_names_out()`.

```python
from deup.core.features import FeaturePipeline, VarianceFeature, SeenBit

pipe = FeaturePipeline([
    ("var", VarianceFeature(method="ensemble")),
    ("seen", SeenBit()),
])
```

## Torch-dependent methods

`DensityFeature(method="flow")` and `VarianceFeature(method="gp")` require
`pip install "deup[torch]"`. Without torch, construction raises `ImportError` with an
install hint; the module still imports cleanly on a torch-free install.

## v0.1 vs v0.2

**v0.1 (this release):** feature builders + pipeline are available as primitives.
`DEUPRegressor` still trains $g$ on raw $X$ by default.

**v0.2 (P6):** `ErrorEstimator` wires `FeaturePipeline` into the DEUP training loop
with target transforms and non-negativity clipping.
