"""Shared benchmark utilities (seeds, paths, metrics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BENCHMARKS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
DEFAULT_SEED = 42


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def write_json(name: str, payload: dict[str, Any]) -> Path:
    out = ensure_results_dir() / name
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def spearman_unc_vs_sqerr(
    unc: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    realized = (y_true - y_pred) ** 2
    rho = spearmanr(unc, realized).statistic
    return float(rho) if rho is not None else float("nan")


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))
