"""deup: Direct Epistemic Uncertainty Prediction for any scikit-learn-style model.

DEUP (Lahlou et al., 2023, TMLR) estimates *epistemic* uncertainty by training a
secondary "error predictor" on a base model's out-of-sample errors, then (optionally)
subtracting an estimate of aleatoric noise. This package provides a maintained,
scikit-learn-compatible implementation with first-class support for time-series and
cross-sectional workflows, where correct out-of-fold error construction (no leakage)
is the difference between a valid and an invalid uncertainty estimate.

This is the library implementation; the DEUP *method* is due to
Lahlou, Jain, Nekoei, Butoi, Bertin, Rector-Brooks, Korablyov, and Bengio (2023).
"""

from __future__ import annotations

from deup.estimators import DEUPClassifier, DEUPRanker, DEUPRegressor

__all__ = ["DEUPClassifier", "DEUPRanker", "DEUPRegressor", "__version__"]
__version__ = "0.3.2"
