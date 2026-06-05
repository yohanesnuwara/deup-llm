"""Feature builders for the DEUP error predictor g(x).

Each builder is a scikit-learn ``TransformerMixin`` that fits on training data only.
They implement the stationarizing features phi_{z^N}(x) from Lahlou et al. (2023),
Sec. 3.2 — density, variance, seen-bit, distance, and residual proxies.

See ``docs/theory.md`` and ``docs/features.md`` for the mathematical definitions.
"""

from __future__ import annotations

from deup.core.features.density import DensityFeature
from deup.core.features.distance import DistanceToTrain
from deup.core.features.pipeline import FeaturePipeline
from deup.core.features.raw import RawFeatures
from deup.core.features.residual import ResidualMagnitude
from deup.core.features.seen_bit import SeenBit
from deup.core.features.variance import VarianceFeature

__all__ = [
    "DensityFeature",
    "DistanceToTrain",
    "FeaturePipeline",
    "RawFeatures",
    "ResidualMagnitude",
    "SeenBit",
    "VarianceFeature",
]
