# Changelog

## [Unreleased]

## [0.4.1] — 2026-06-05

### Changed

- PyPI short description mentions gradient-boosting backends.
- Docs: remove completed GBM presets from “future work” sections.

## [0.4.0] — 2026-06-05

### Added

- **Tabular GBM backends (P16):** `TabularDEUP(backend="lgbm"|"xgb"|"catboost")` wires
  LightGBM, XGBoost, and CatBoost as default base + error predictors.
- Optional extras: `deup[xgb]`, `deup[catboost]`, `deup[gbm-all]`.
- Gate tests (`tests/test_tabular_backends.py`) + CI job for all GBM extras.
- Regression benchmark optional rows for GBM tabular presets.

## [0.3.2] — 2026-06-05

### Added

- **TorchUncertainty integration (P15):** upstream DEUP post-processing PR
  ([#313](https://github.com/torch-uncertainty/torch-uncertainty/pull/313));
  docs page for PyTorch / Lightning workflows.

## [0.3.1] — 2026-06-05

### Added

- **Community & release (P14):** `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue/PR
  templates, `LAUNCH.md` draft, release workflow smoke-install + dry-run dispatch.

### Changed

- Public docs no longer assume private thesis context; migration page removed from site nav.
- `RELEASING.md` expanded with troubleshooting for early failed `v0.1.0` deploys.

## [0.3.0] — 2026-06-05

### Added

- **Documentation (P13):** five-axis conceptual guide, four tutorials (tabular, finance,
  classification+conformal, active learning), README comparison table, CI tutorial smoke
  tests + `mkdocs build --strict` in CI.
- **Walk-forward finance helper:** `walkforward_g_on_enriched` for pre-computed residual panels.
- **Benchmark suite (P12):** N-sweep + figure, regression comparison (DEUP vs
  ensemble/conformal/Laplace), CIFAR proxy, finance walk-forward; committed results.

### Added (0.3.0 core)

- **Reliability diagnostics** (`deup.diagnostics`): `AggregationReliability` /
  `should_trust_aggregate` and pluggable `HealthIndex`.
- **Domain presets** (`deup.domains`): `CrossSectionalDEUP`, `TabularDEUP`, `VisionDEUP`.

## [0.2.0] — 2026-06-05

### Added

- **`DEUPClassifier`** — classification with log-loss / Brier OOF errors + `predict_proba`
- **`DEUPRanker`** — cross-sectional ranking; `loss="rank"`, `PurgedWalkForward` default,
  rank-geometry residualization ON by default
- **`acquire(pool, k)`** — active-learning hook (top-k by epistemic uncertainty)
- Refactored **`DEUPRegressor`** onto `ErrorEstimator` + optional `features` /
  `aleatoric` / `decompose`

## [0.1.1] — 2026-06-04

First release published to PyPI.

### Fixed

- `OOFErrorCollector` multiclass `predict_proba` support and overlap-fold guard.

### Added

- Research-grade docstrings documenting the OOF refit assumption (DEUP Algorithm 2).

## [0.1.0] — 2026-06-04

First public release (PyPI publish failed — trusted publishing not yet configured;
superseded by v0.1.1).

### Added

- `DEUPRegressor`, `OOFErrorCollector`, splitters, loss registry, California housing benchmark, MkDocs site.

[0.4.0]: https://github.com/ursinasanderink/deup/releases/tag/v0.4.0
[0.3.2]: https://github.com/ursinasanderink/deup/releases/tag/v0.3.2
[0.3.1]: https://github.com/ursinasanderink/deup/releases/tag/v0.3.1
[0.3.0]: https://github.com/ursinasanderink/deup/releases/tag/v0.3.0
[0.1.1]: https://github.com/ursinasanderink/deup/releases/tag/v0.1.1
[0.1.0]: https://github.com/ursinasanderink/deup/releases/tag/v0.1.0
