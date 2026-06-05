# Decomposition & rank residualization

This page covers the v0.2 components that turn the raw error estimate $g(x)$ into a
reported epistemic signal: the error estimator, aleatoric estimators, the
$\hat{e} = \max(0, g - a)$ decomposition, and cross-sectional rank-geometry
residualization. See [Theory](theory.md) for the underlying math.

## ErrorEstimator

`ErrorEstimator` is the reusable DEUP error model $g$ — feature pipeline +
target transform + non-negativity, fit on out-of-fold errors.

```python
from deup.core import ErrorEstimator
from deup.core.features import DensityFeature, FeaturePipeline, RawFeatures
from deup.core.oof import OOFErrorCollector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

oof = OOFErrorCollector(
    RandomForestRegressor(), cv=KFold(5), loss="squared"
).fit_collect(X, y)

g = ErrorEstimator(
    features=FeaturePipeline([("raw", RawFeatures()), ("density", DensityFeature())]),
    target_transform="log",
).fit(X[oof.indices], oof.errors)

error_estimate = g.predict(X_new)   # >= 0
```

## Aleatoric estimators $a(x)$

Model-agnostic estimates of the irreducible noise floor $A(x) = \mathrm{Var}(Y\mid X=x)$
(variance scale, matching a squared-error target).

| Estimator | $a(x)$ | When |
|---|---|---|
| `Homoscedastic` | constant $\sigma^2$ | noise ~ constant across $\mathcal{X}$ |
| `Heteroscedastic` | local k-NN label variance | input-dependent noise |
| `Quantile` | $((q_{hi}-q_{lo})/z)^2$ from quantile regression | skewed / tail noise |

```python
from deup.core import Heteroscedastic

a = Heteroscedastic(k=20).fit(X, y).predict(X_new)
```

## Decomposition

```python
from deup.core import decompose_epistemic

e_hat = decompose_epistemic(error_estimate, a)   # max(0, g - a)
# a=None -> conservative proxy e_hat = g (the v0.1 default)
```

$\hat{e}$ is always non-negative.

## Rank-geometry residualization (Finding 3)

For cross-sectional rankers, $g$ and the loss target can be partly **mechanical rank
geometry** rather than genuine error. `RankResidualizer` fits an isotonic map from the
within-group rank of $|score|$ to the signal and subtracts it, leaving the part *not*
explained by rank geometry.

```python
from deup.core import RankResidualizer, coupling_retention_report

# decouple g from rank geometry, per date
res = RankResidualizer().fit(g_values, abs_score, groups=dates)
g_decoupled = res.transform(g_values, abs_score, groups=dates)

# diagnostics: coupling before/after + loss-association retention
report = coupling_retention_report(g_values, score, loss, groups=dates)
print(report.coupling_before, report.coupling_after, report.retention)
```

!!! note "Thesis finding"
    Residualization decoupled the signal (per-date $\rho(\hat{e}, |score|)$:
    $0.616 \to 0.317$) while **retaining ~92.5%** of the loss association. This is
    **off by default** and **on in `DEUPRanker`** (P7).

## Density kill criterion (Finding 3 corollary)

Density features can be an **informative null** in homogeneous universes. The kill
criterion drops them when their gain importance is negligible **and** they barely move
the loss partial-correlation.

```python
from deup.core import density_kill_criterion

decision = density_kill_criterion(gain_importance=1e-5, delta_partial_corr=0.001)
print(decision.keep, decision.reason)   # False, "killed: ..."
```

Use `partial_correlation(a, b, control)` to compute the $\Delta$ partial-correlation
with vs without the density feature.
