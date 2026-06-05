# Conformal calibration

DEUP's `predict_epistemic` returns an *uncalibrated* score: higher means "less
trustworthy", but not a probability. **Split-conformal calibration** turns it into
prediction intervals with finite-sample, distribution-free marginal coverage
$P(y \in [\hat{y}^-, \hat{y}^+]) \ge 1 - \alpha$ — using the DEUP signal as the
interval's *width*.

## How it works

On a **held-out** calibration set, compute normalized residuals
$r_i = |y_i - f(x_i)| / g(x_i)$ and take their $(1-\alpha)$ empirical quantile $q$.
The interval at a new point is

$$
[\,f(x) - q\,g(x),\;\; f(x) + q\,g(x)\,].
$$

Intervals are **narrow where $g$ is small** (confident) and wide where $g$ is large —
locally adaptive coverage, unlike a constant-width baseline.

## Usage

```python
from deup import DEUPRegressor

model = DEUPRegressor(base_model=my_model).fit(X_train, y_train)

# calibrate on a separate held-out split (NOT the training data)
model.calibrate(X_cal, y_cal, method="normalized", alpha=0.1)

interval = model.predict_interval(X_test)
interval.lower, interval.upper, interval.width
```

!!! warning "Use held-out data"
    Coverage guarantees require the calibration set to be unseen by both the base model
    $f$ and the error model $g$. Don't calibrate on training rows.

## Methods

| `method` | Score | Use when |
|---|---|---|
| `normalized` (default) | $\lvert y-f(x)\rvert / g(x)$ | locally adaptive intervals |
| `mondrian` | per-group quantile | group/regime-conditional coverage |
| `cqr` | conformalized quantile regression | you already have quantile models |

```python
# Mondrian: group-conditional coverage (e.g. per regime)
model.calibrate(X_cal, y_cal, method="mondrian", alpha=0.1, groups=regime_cal)
interval = model.predict_interval(X_test, groups=regime_test)
```

The standalone `UncertaintyCalibrator` works with raw arrays (any base model):

```python
from deup.calibration import UncertaintyCalibrator

cal = UncertaintyCalibrator(method="normalized", alpha=0.1)
cal.fit(y_cal, y_pred_cal, uncertainty_cal)
interval = cal.predict_interval(y_pred_test, uncertainty_test)
```

## MAPIE interop

`deup` is **complementary** to [MAPIE](https://mapie.readthedocs.io/): MAPIE supplies
mature conformal machinery, DEUP supplies a high-quality per-point scale $g(x)$. Expose
the DEUP scale as a normalizer:

```python
from deup.calibration import deup_normalizer

normalizer = deup_normalizer(model)   # .predict(X) == model.predict_epistemic(X)
scale = normalizer.predict(X_cal)     # feed into MAPIE as a residual scale
```

See [`examples/mapie_interop.py`](https://github.com/ursinasanderink/deup/blob/main/examples/mapie_interop.py)
for a runnable script.

## Coverage guarantee

Split conformal gives the finite-sample bound (Lei et al., 2018)

$$
1 - \alpha \;\le\; P(y \in \hat{C}(x)) \;\le\; 1 - \alpha + \tfrac{1}{n_{\text{cal}}+1},
$$

so intervals may *slightly over-cover*; this is correct, not a bug. `deup`'s test suite
checks empirical coverage within tolerance on i.i.d. and purged time-split fixtures.
