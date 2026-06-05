"""P10: domain presets (finance, tabular, vision) — presets only, no logic duplication."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from deup.domains.finance import CrossSectionalDEUP, enrich_panel
from deup.domains.tabular import TabularDEUP, tabular_feature_pipeline
from deup.domains.vision import EmbeddingUncertaintyFeatures, VisionDEUP


def _make_panel(n_dates: int = 18, n_assets: int = 35, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for d in range(n_dates):
        for _ in range(n_assets):
            score = float(rng.normal())
            rows.append(
                {
                    "date": d,
                    "score": score,
                    "vol_20d": float(rng.uniform(0.1, 0.5)),
                    "vol_60d": float(rng.uniform(0.1, 0.5)),
                    "market_regime_enc": float(rng.choice([-1.0, 0.0, 1.0])),
                    "market_vol_21d": float(rng.uniform(0.1, 0.3)),
                    "vix_percentile_252d": float(rng.uniform(0, 1)),
                    "target_20d": score + float(rng.normal(scale=0.2)),
                    "target_60d": score + float(rng.normal(scale=0.35)),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ finance gate
def test_cross_sectional_deup_panel_end_to_end() -> None:
    panel = _make_panel()
    n_dates = panel["date"].nunique()
    split = int(n_dates * 0.6)
    train = panel[panel["date"] < split].copy()
    cal = panel[(panel["date"] >= split) & (panel["date"] < split + 3)].copy()
    test = panel[panel["date"] >= split + 3].copy()

    model = CrossSectionalDEUP(cv=3, embargo=1, horizon=20, random_state=0).fit(train)
    model.calibrate(cal, alpha=0.1)

    unc = model.predict_epistemic(test)
    assert unc.shape == (len(test),)
    assert np.all(unc >= 0.0)

    interval = model.predict_interval(test)
    assert interval.lower.shape == interval.upper.shape == (len(test),)

    report = model.health_report(test)
    assert report.health.shape == (test["date"].nunique(),)
    assert report.gate.shape == report.health.shape
    assert isinstance(report.verdict(report.labels[0]), bool)

    # Rank-decoupled: residualization enabled and applied at fit time (Finding 3).
    assert model.ranker_.residualize_rank is True
    assert model.ranker_._residualizer_ is not None


def test_cross_sectional_horizon_selects_target_column() -> None:
    panel = _make_panel(n_dates=12, n_assets=20)
    m20 = CrossSectionalDEUP(cv=2, horizon=20, random_state=0).fit(panel)
    m60 = CrossSectionalDEUP(cv=2, horizon=60, random_state=0).fit(panel)
    assert m20.target_col_ == "target_20d"
    assert m60.target_col_ == "target_60d"


def test_enrich_panel_adds_derived_columns() -> None:
    df = pd.DataFrame({"date": [0, 0, 1, 1], "score": [1.0, -2.0, 0.5, -0.5]})
    out = enrich_panel(df)
    assert "abs_score" in out.columns
    assert "cross_sectional_rank" in out.columns


# ------------------------------------------------------------------ tabular preset
def test_tabular_deup_regression_with_density_features() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = X @ np.array([1.0, -0.5, 0.2, 0.0, 0.1]) + rng.normal(scale=0.3, size=300)
    model = TabularDEUP(cv=4, random_state=0).fit(X, y)
    unc = model.predict_epistemic(X)
    assert unc.shape == (300,)
    assert np.all(unc >= 0.0)


def test_tabular_feature_pipeline_has_density() -> None:
    pipe = tabular_feature_pipeline()
    assert any(name == "density" for name, _ in pipe.transformers)


# ------------------------------------------------------------------ vision preset
def test_vision_deup_embedding_density_variance_flow() -> None:
    rng = np.random.default_rng(2)
    # Tiny "image" tensors: (n, channels, h, w)
    X = rng.normal(size=(80, 3, 8, 8))
    y = (rng.normal(size=80) > 0).astype(int)
    model = VisionDEUP(
        base_model=LogisticRegression(max_iter=500),
        cv=3,
        random_state=0,
    ).fit(X, y)
    unc = model.predict_epistemic(X)
    assert unc.shape == (80,)
    assert np.all(unc >= 0.0)
    assert np.all(np.isfinite(unc))


def test_embedding_uncertainty_features_on_flat_inputs() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 10))
    y = rng.normal(size=60)
    feat = EmbeddingUncertaintyFeatures(random_state=0).fit(X, y)
    out = feat.transform(X)
    assert out.shape == (60, 2)
    assert np.all(np.isfinite(out))


# ------------------------------------------------------------------ no duplication
def test_domain_modules_are_preset_thin_wrappers() -> None:
    """Domain modules delegate to core estimators — no duplicated OOF/g logic."""
    root = Path(__file__).resolve().parents[1] / "src" / "deup" / "domains"
    for name in ("finance.py", "tabular.py", "vision.py"):
        text = (root / name).read_text()
        assert "OOFErrorCollector" not in text
        assert "class ErrorEstimator" not in text
    finance_src = inspect.getsource(CrossSectionalDEUP)
    assert "DEUPRanker" in finance_src
    assert "PurgedWalkForward" in finance_src
    assert "HealthIndex" in finance_src
    assert "UncertaintyCalibrator" not in finance_src  # uses ranker.calibrate


def test_finance_requires_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    import deup.domains.finance as fin

    monkeypatch.setattr(fin, "pd", None)
    with pytest.raises(ImportError, match="deup\\[finance\\]"):
        CrossSectionalDEUP().fit(pd.DataFrame({"date": [0], "y": [1.0]}))
