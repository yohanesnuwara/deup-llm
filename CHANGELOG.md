# Changelog

## [Unreleased]

## [0.2.0] — 2026-06-05

### Added

- **`DEUPClassifier`** — classification with log-loss / Brier OOF errors + `predict_proba`
- **`DEUPRanker`** — cross-sectional ranking; `loss="rank"`, `PurgedWalkForward` default,
  rank-geometry residualization ON by default (Finding 3)
- **`acquire(pool, k)`** — active-learning hook (top-k by epistemic uncertainty)
- Refactored **`DEUPRegressor`** onto `ErrorEstimator` + optional `features` /
  `aleatoric` / `decompose`
- Docs updated for all three estimators and `acquire`

## [0.1.1] — 2026-06-04

First release published to PyPI.

### Fixed

- `OOFErrorCollector` now supports multiclass `predict_proba` targets (previously
  only binary worked; multiclass stored 2-D probabilities and crashed).
- Guard against rows assigned to multiple test folds (e.g. repeated CV): a warning
  is raised and one error per row is kept, preserving honest OOF targets.
- Validate `groups` length against `n_rows` and the loss output length.

### Added

- Research-grade docstrings documenting the "g trained on a slightly smaller f"
  refit assumption (DEUP Algorithm 2) plus a "How it works" docs section.

## [0.1.0] — 2026-06-04

First public release.

### Added

- `DEUPRegressor` — sklearn-compatible wrapper with `predict(..., return_uncertainty=True)`
- Leakage-correct `OOFErrorCollector` (DEUP Algorithm 2 / K-fold OOF errors)
- Splitters: `PurgedWalkForward`, re-export `KFold` / `TimeSeriesSplit`
- Loss registry: `squared`, `absolute`, `logloss`, `brier`, `pinball`, `rank`
- Target transforms: `log`, `asinh`, `none` for error-predictor training
- Benchmark: DEUP vs ensemble vs conformal on California housing
- MkDocs documentation site
- 54+ unit tests including parity-exact OOF and leakage gate

### Notes

- Aleatoric decomposition (`ê = max(0, g - a)`), conformal intervals, and
  `DEUPClassifier` / `DEUPRanker` are planned for v0.2.

[0.1.1]: https://github.com/ursinasanderink/deup/releases/tag/v0.1.1
[0.1.0]: https://github.com/ursinasanderink/deup/releases/tag/v0.1.0
