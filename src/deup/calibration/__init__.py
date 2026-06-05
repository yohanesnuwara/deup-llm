"""Conformal calibration for DEUP uncertainty.

Turns the (uncalibrated) epistemic score ``g(x)`` into prediction intervals with
finite-sample, distribution-free marginal coverage via split conformal prediction.
The DEUP signal enters as the **normalization** of the conformity score, so intervals
are narrow where the model is confident and wide where ``g`` is large.
"""

from __future__ import annotations

from deup.calibration.conformal import (
    ConformalResult,
    UncertaintyCalibrator,
    deup_normalizer,
)

__all__ = [
    "ConformalResult",
    "UncertaintyCalibrator",
    "deup_normalizer",
]
