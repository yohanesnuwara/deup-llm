"""Core protocols, typed result containers, and the grouped/panel data model.

These are the framework-agnostic foundations every estimator in ``deup`` is built
on. Nothing here imports a heavy backend (no torch, lightgbm, or pandas); the only
runtime dependency is numpy.
"""

from __future__ import annotations

from deup.core.aleatoric import Heteroscedastic, Homoscedastic, Quantile
from deup.core.decompose import (
    CouplingReport,
    DensityKillDecision,
    RankResidualizer,
    coupling_retention_report,
    decompose_epistemic,
    density_kill_criterion,
    partial_correlation,
)
from deup.core.error_estimator import ErrorEstimator
from deup.core.grouping import Grouping
from deup.core.losses import (
    TargetTransform,
    apply_error_transform,
    get_loss,
    inverse_error_transform,
)
from deup.core.oof import OOFErrorCollector
from deup.core.protocols import Predictor, ProbabilisticPredictor
from deup.core.types import OOFResult, UncertaintyResult

__all__ = [
    "CouplingReport",
    "DensityKillDecision",
    "ErrorEstimator",
    "Grouping",
    "Heteroscedastic",
    "Homoscedastic",
    "OOFErrorCollector",
    "OOFResult",
    "Predictor",
    "ProbabilisticPredictor",
    "Quantile",
    "RankResidualizer",
    "TargetTransform",
    "UncertaintyResult",
    "apply_error_transform",
    "coupling_retention_report",
    "decompose_epistemic",
    "density_kill_criterion",
    "get_loss",
    "inverse_error_transform",
    "partial_correlation",
]
