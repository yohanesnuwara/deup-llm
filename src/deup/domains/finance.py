"""Cross-sectional finance / time-series ranking preset (flagship).

``CrossSectionalDEUP`` wires :class:`~deup.estimators.DEUPRanker` with
:class:`~deup.splitters.PurgedWalkForward`, rank-geometry residualization,
vol/breadth/regime feature columns, and a default :class:`~deup.diagnostics.HealthIndex`
for context-level gating in the low-N / non-i.i.d. regime (Finding 2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from deup.core.decompose import coupling_retention_report
from deup.core.features import FeaturePipeline, RawFeatures
from deup.diagnostics import HealthIndex, HealthReport
from deup.estimators import DEUPRanker
from deup.splitters import PurgedWalkForward

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# Thesis-aligned g-feature preset (subset used when columns are present).
FINANCE_G_FEATURES: tuple[str, ...] = (
    "score",
    "abs_score",
    "vol_20d",
    "vol_60d",
    "mom_1m",
    "adv_20d",
    "vix_percentile_252d",
    "market_regime_enc",
    "market_vol_21d",
    "market_return_21d",
    "cross_sectional_rank",
    "credit_ratio",
    "breadth_ratio",
    "downside_rv_share",
)


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError(
            "CrossSectionalDEUP requires pandas. Install with: pip install deup[finance]"
        )
    return pd


def enrich_panel(df: Any, *, date_col: str = "date", score_col: str = "score") -> Any:
    """Add derived rank/regime columns expected by the finance g-feature preset."""
    _require_pandas()
    out = df.copy()
    if score_col in out.columns:
        out["abs_score"] = out[score_col].abs()
        out["cross_sectional_rank"] = out.groupby(date_col)[score_col].rank(pct=True)
    if "market_regime" in out.columns and "market_regime_enc" not in out.columns:
        regime_map = {"bull": 1, "1": 1, "neutral": 0, "0": 0, "bear": -1, "-1": -1}
        out["market_regime_enc"] = (
            out["market_regime"].astype(str).map(regime_map).fillna(0).astype(float)
        )
    return out


def resolve_finance_features(
    df: Any,
    *,
    date_col: str,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> list[str]:
    """Return feature column names: explicit list, preset intersection, or all numeric."""
    if feature_cols is not None:
        return list(feature_cols)
    preset = [c for c in FINANCE_G_FEATURES if c in df.columns]
    if preset:
        return preset
    skip = {date_col, target_col}
    return [c for c in df.select_dtypes(include="number").columns if c not in skip]


def finance_feature_pipeline() -> FeaturePipeline:
    """Build a raw-feature pipeline (panel columns are selected upstream)."""
    return FeaturePipeline([("panel", RawFeatures())])


class CrossSectionalDEUP(BaseEstimator):
    """Panel-data preset for cross-sectional stock rankers.

    Defaults: ``PurgedWalkForward`` + rank residualization + finance g-features +
    ``HealthIndex`` for per-date context gating.

    Parameters
    ----------
    base_model:
        Primary ranker (defaults to HGB regressor inside :class:`DEUPRanker`).
    date_col:
        Column holding the cross-section date / group label.
    target_col:
        Column holding the ranking target. If ``horizon`` is set, defaults to
        ``f"target_{horizon}d"`` when that column exists.
    horizon:
        Optional return horizon in days; selects ``target_{horizon}d`` when present.
    feature_cols:
        Explicit g-feature columns; otherwise uses :data:`FINANCE_G_FEATURES` present
        in the panel.
    cv, embargo:
        Walk-forward splitter settings (``PurgedWalkForward`` when ``cv`` is int).
    health_index:
        Context health scorer (default: three-component :class:`HealthIndex`).
    """

    def __init__(
        self,
        base_model: Any = None,
        *,
        date_col: str = "date",
        target_col: str | None = None,
        horizon: int | None = None,
        feature_cols: list[str] | None = None,
        cv: Any = 5,
        embargo: int = 1,
        health_index: HealthIndex | None = None,
        random_state: int | None = None,
    ) -> None:
        self.base_model = base_model
        self.date_col = date_col
        self.target_col = target_col
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.cv = cv
        self.embargo = embargo
        self.health_index = health_index if health_index is not None else HealthIndex()
        self.random_state = random_state

    def _resolve_target_col(self, df: Any) -> str:
        if self.target_col is not None:
            return self.target_col
        if self.horizon is not None:
            col = f"target_{self.horizon}d"
            if col in df.columns:
                return col
        if "y" in df.columns:
            return "y"
        raise ValueError("Could not infer target column; pass target_col= or horizon=.")

    def _resolve_cv(self) -> Any:
        if isinstance(self.cv, int):
            return PurgedWalkForward(n_splits=self.cv, embargo=self.embargo)
        return self.cv

    def _build_ranker(self) -> DEUPRanker:
        return DEUPRanker(
            base_model=self.base_model,
            features=finance_feature_pipeline(),
            cv=self._resolve_cv(),
            residualize_rank=True,
            random_state=self.random_state,
        )

    def fit(self, panel_df: Any, y: npt.ArrayLike | None = None) -> CrossSectionalDEUP:
        """Fit on a long-format panel DataFrame."""
        _require_pandas()
        df = enrich_panel(panel_df, date_col=self.date_col)
        target_col = self._resolve_target_col(df)
        feat_cols = resolve_finance_features(
            df, date_col=self.date_col, target_col=target_col, feature_cols=self.feature_cols
        )
        X = df[feat_cols].to_numpy(dtype=float)
        y_arr = df[target_col].to_numpy(dtype=float) if y is None else np.asarray(y, dtype=float)
        groups = df[self.date_col].to_numpy()

        self.feature_cols_ = feat_cols
        self.target_col_ = target_col
        self.ranker_ = self._build_ranker()
        self.ranker_.fit(X, y_arr, groups=groups)

        ref_col = "market_vol_21d" if "market_vol_21d" in df.columns else feat_cols[0]
        ref = df[ref_col].to_numpy(dtype=float)
        self._reference_feature_ = ref[: max(len(ref) // 4, 1)]
        return self

    def predict_epistemic(self, panel_df: Any) -> npt.NDArray[Any]:
        """Rank-residualized epistemic uncertainty per row."""
        X, groups = self._panel_xy(panel_df)
        return self.ranker_.predict_epistemic(X, groups=groups)

    def predict(
        self, panel_df: Any, *, return_uncertainty: bool = False
    ) -> npt.NDArray[Any] | tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        X, groups = self._panel_xy(panel_df)
        return self.ranker_.predict(X, return_uncertainty=return_uncertainty, groups=groups)

    def calibrate(self, panel_df: Any, *, alpha: float = 0.1) -> CrossSectionalDEUP:
        """Conformal-calibrate on a held-out panel split (separate from fit)."""
        X, groups = self._panel_xy(panel_df)
        y = panel_df[self.target_col_].to_numpy(dtype=float)
        self.ranker_.calibrate(X, y, alpha=alpha, groups=groups)
        return self

    def predict_interval(self, panel_df: Any) -> Any:
        X, groups = self._panel_xy(panel_df)
        return self.ranker_.predict_interval(X, groups=groups)

    def health_report(self, panel_df: Any) -> HealthReport:
        """Per-date composite context health (Finding 2 remedy)."""
        df = enrich_panel(panel_df, date_col=self.date_col)
        X, groups = self._panel_xy(df)
        y = df[self.target_col_].to_numpy(dtype=float)
        pred = np.asarray(self.ranker_.base_model_.predict(X), dtype=float)
        if "rank_loss" in df.columns:
            loss = df["rank_loss"].to_numpy(dtype=float)
        else:
            loss = np.abs(y - pred)
        ref_col = "market_vol_21d" if "market_vol_21d" in df.columns else self.feature_cols_[0]
        feature = df[ref_col].to_numpy(dtype=float)
        disagreement = (
            df["vol_20d"].to_numpy(dtype=float)
            if "vol_20d" in df.columns
            else np.full(len(df), float(np.std(pred)))
        )
        return self.health_index.compute(
            groups,
            {
                "loss": loss,
                "feature": feature,
                "feature_reference": self._reference_feature_,
                "disagreement": disagreement,
            },
        )

    def rank_coupling_report(self, panel_df: Any) -> Any:
        """Diagnostic: rank-geometry coupling before/after residualization."""
        df = enrich_panel(panel_df, date_col=self.date_col)
        X, groups = self._panel_xy(df)
        g_raw = self.ranker_.error_estimator_.predict(X)
        scores = np.abs(df["score"].to_numpy(dtype=float)) if "score" in df.columns else g_raw
        y = df[self.target_col_].to_numpy(dtype=float)
        loss = (
            df["rank_loss"].to_numpy(dtype=float)
            if "rank_loss" in df.columns
            else np.abs(y - np.asarray(self.ranker_.base_model_.predict(X), dtype=float))
        )
        return coupling_retention_report(g_raw, scores, loss, groups=groups)

    def _panel_xy(self, panel_df: Any) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        _require_pandas()
        df = enrich_panel(panel_df, date_col=self.date_col)
        X = df[self.feature_cols_].to_numpy(dtype=float)
        groups = df[self.date_col].to_numpy()
        return X, groups
