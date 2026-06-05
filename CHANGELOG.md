# Changelog

## [Unreleased]

### Added

- **P5 feature builders** for `g(x)`: `RawFeatures`, `DensityFeature`
  (`mahalanobis`/`knn`/`kde`), `VarianceFeature` (ensemble), `DistanceToTrain`,
  `SeenBit`, `ResidualMagnitude`, and `FeaturePipeline`.
- **P6 decomposition**: `ErrorEstimator` (reusable `g`), aleatoric estimators
  (`Homoscedastic`, `Heteroscedastic`, `Quantile`), `decompose_epistemic`
  (`max(0, g - a)`), `RankResidualizer` + `coupling_retention_report` for
  rank-geometry decoupling (Finding 3), and `density_kill_criterion`.
- Docs: `theory.md` (math), `features.md`, `decomposition.md` with MathJax.

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
