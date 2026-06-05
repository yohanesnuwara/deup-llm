"""P5: feature builders for g(x) and FeaturePipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from deup.core.features import (
    DensityFeature,
    DistanceToTrain,
    FeaturePipeline,
    RawFeatures,
    ResidualMagnitude,
    SeenBit,
    VarianceFeature,
)


def _toy_xy(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.0, -0.5, 0.2, 0.0]) + rng.normal(scale=0.3, size=n)
    return X, y


def test_raw_features_passthrough() -> None:
    X, _ = _toy_xy()
    out = RawFeatures().fit_transform(X)
    assert out.shape == X.shape
    assert np.allclose(out, X)


def test_mahalanobis_matches_thesis_gaussian_density() -> None:
    """Parity with thesis ``GaussianDensity.log_prob`` (diagonal Gaussian)."""
    X, _ = _toy_xy(n=200, seed=42)
    feat = DensityFeature(method="mahalanobis", var_floor=1e-6).fit(X)
    log_q = feat.transform(X).ravel()

    mu = X.mean(axis=0)
    sigma2 = np.maximum(X.var(axis=0), 1e-6)
    log_sigma2 = np.log(sigma2)
    diff2 = (X - mu) ** 2 / sigma2
    d = X.shape[1]
    expected = -0.5 * (diff2 + log_sigma2).sum(axis=1) - 0.5 * d * math.log(2 * math.pi)
    assert np.allclose(log_q, expected, rtol=1e-10)


def test_density_knn_and_kde_shapes() -> None:
    X, _ = _toy_xy()
    knn = DensityFeature(method="knn", k=3).fit_transform(X)
    kde = DensityFeature(method="kde", bandwidth=0.5).fit_transform(X)
    assert knn.shape == (len(X), 1)
    assert kde.shape == (len(X), 1)
    assert np.all(np.isfinite(knn))
    assert np.all(np.isfinite(kde))


def test_flow_requires_torch() -> None:
    X, _ = _toy_xy(n=30)
    feat = DensityFeature(method="flow")
    try:
        import torch  # noqa: F401

        with pytest.raises(NotImplementedError):
            feat.fit(X)
    except ImportError:
        with pytest.raises(ImportError, match=r"deup\[torch\]"):
            feat.fit(X)


def test_variance_ensemble() -> None:
    X, y = _toy_xy()
    out = VarianceFeature(
        method="ensemble",
        estimator=LinearRegression(),
        n_estimators=5,
        random_state=0,
    ).fit_transform(X, y)
    assert out.shape == (len(X), 1)
    assert np.all(np.isfinite(out))


def test_gp_requires_torch() -> None:
    X, y = _toy_xy(n=30)
    feat = VarianceFeature(method="gp")
    try:
        import torch  # noqa: F401

        with pytest.raises(NotImplementedError):
            feat.fit(X, y)
    except ImportError:
        with pytest.raises(ImportError, match=r"deup\[torch\]"):
            feat.fit(X, y)


def test_distance_to_train() -> None:
    X, _ = _toy_xy()
    dist = DistanceToTrain(k=3).fit_transform(X)
    assert dist.shape == (len(X), 1)
    assert np.all(dist >= 0)


def test_seen_bit_exact_duplicates() -> None:
    X, _ = _toy_xy(n=50)
    X_dup = np.vstack([X, X[0:3]])
    seen = SeenBit().fit(X).transform(X_dup).ravel()
    assert seen[:50].sum() >= 50  # all training rows seen
    assert np.all(seen[50:] == 1.0)


def test_residual_magnitude_knn_smoothing() -> None:
    X, y = _toy_xy()
    est = LinearRegression()
    out = ResidualMagnitude(est, k=5).fit_transform(X, y)
    assert out.shape == (len(X), 1)
    assert np.all(out >= 0)


def test_feature_pipeline_composes_three_builders() -> None:
    X, y = _toy_xy()
    pipe = FeaturePipeline(
        [
            ("raw", RawFeatures()),
            ("density", DensityFeature(method="mahalanobis")),
            ("dist", DistanceToTrain(k=2)),
            ("seen", SeenBit()),
        ]
    )
    out = pipe.fit_transform(X, y)
    assert out.shape == (len(X), X.shape[1] + 3)
    names = pipe.get_feature_names_out()
    assert len(names) == out.shape[1]
    assert any("log_density" in str(n) for n in names)


def test_features_import_without_torch() -> None:
    """Torch-free install must import the features module."""
    from deup.core import features  # noqa: F401

    assert DensityFeature is not None
