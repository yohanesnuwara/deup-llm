# Domain presets

The core library is domain-agnostic; these modules are **thin presets** that wire the
right splitter, features, and diagnostics for common workflows. They do not duplicate
OOF collection or error-estimator logic — see ``ARCHITECTURE.md``.

## Cross-sectional finance (flagship)

```python
import pandas as pd
from deup.domains.finance import CrossSectionalDEUP

# long-format panel: one row per (date, asset)
panel = pd.read_parquet("signals.parquet")  # columns: date, score, vol_20d, ...

model = CrossSectionalDEUP(horizon=20, cv=5, embargo=1).fit(panel)
model.calibrate(cal_panel, alpha=0.1)

pred, unc = model.predict(test_panel, return_uncertainty=True)
health = model.health_report(test_panel)   # per-date context gating (Finding 2)
health.gate                                # bool per date: trust / trade?
```

Defaults wired in:

| Setting | Value |
|---|---|
| Estimator | :class:`~deup.estimators.DEUPRanker` |
| CV | :class:`~deup.splitters.PurgedWalkForward` + embargo |
| Rank geometry | residualization **ON** (Finding 3) |
| g-features | vol / breadth / regime preset columns when present |
| Context health | :class:`~deup.diagnostics.HealthIndex` |

Requires ``pip install "deup[finance]"`` (pandas).

## Generic tabular

```python
from deup.domains.tabular import TabularDEUP

model = TabularDEUP(task="regression", cv=5).fit(X, y)
unc = model.predict_epistemic(X_test)
```

Wires ``KFold`` + raw ``X`` + Mahalanobis density features for ``g``.

## Vision / OOD classification

```python
from deup.domains.vision import VisionDEUP

model = VisionDEUP(cv=5).fit(images, labels)   # tensors OK — auto-flattened
unc = model.predict_epistemic(images)
```

Wires embedding → density + variance features for ``g`` (CIFAR-style high-N path).
Pass a custom ``embedding=`` transformer or callable for CNN embeddings.
