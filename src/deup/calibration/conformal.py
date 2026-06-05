"""Split-conformal calibration with DEUP-normalized scores.

Given a base prediction ``f(x)`` and a positive uncertainty scale ``u(x)`` (here the
DEUP epistemic estimate ``g(x)``), split-conformal prediction (Lei et al., 2018;
Romano et al., 2019) yields intervals with finite-sample marginal coverage

    P(y in [f(x) - q*u(x), f(x) + q*u(x)]) >= 1 - alpha

where ``q`` is the ``(1-alpha)`` empirical quantile of the **normalized residuals**
``r_i = |y_i - f(x_i)| / u(x_i)`` on a held-out calibration set.

Three methods:

- ``normalized`` — single global quantile of normalized residuals (locally adaptive
  via ``u(x)``; this is the DEUP-normalized interval).
- ``mondrian`` — a separate quantile **per group** (e.g. per regime / per date bucket),
  giving group-conditional coverage (Vovk, 2012).
- ``cqr`` — Conformalized Quantile Regression (Romano et al., 2019): calibrate
  pre-fit lower/upper quantile predictions; ``u`` is not used.

The calibrator is model-agnostic: it consumes arrays, so it works with any base model
and any uncertainty scale, and is wired into the DEUP estimators via ``predict_interval``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

ConformalMethod = Literal["normalized", "mondrian", "cqr"]


def _finite_sample_quantile_level(n: int, alpha: float) -> float:
    """The conformal quantile level ``ceil((n+1)(1-alpha))/n`` (clipped to [0, 1])."""
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    return float(min(max(level, 0.0), 1.0))


@dataclass(frozen=True)
class ConformalResult:
    """Lower/upper prediction-interval bounds."""

    lower: npt.NDArray[Any]
    upper: npt.NDArray[Any]

    @property
    def width(self) -> npt.NDArray[Any]:
        return np.asarray(self.upper - self.lower, dtype=float)


class UncertaintyCalibrator:
    """Split-conformal calibration normalized by a DEUP uncertainty scale.

    Parameters
    ----------
    method:
        ``"normalized"`` (default), ``"mondrian"``, or ``"cqr"``.
    alpha:
        Miscoverage level; target coverage is ``1 - alpha``.
    eps:
        Floor added to the uncertainty scale to avoid division by zero.

    Notes
    -----
    Fit on a **held-out calibration set** that the base model and ``g`` did not train
    on, otherwise coverage guarantees do not hold. The DEUP estimators handle this by
    calibrating on out-of-fold predictions.
    """

    def __init__(
        self,
        method: ConformalMethod = "normalized",
        *,
        alpha: float = 0.1,
        eps: float = 1e-9,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.method = method
        self.alpha = alpha
        self.eps = eps

    # ----------------------------------------------------------------- fit
    def fit(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        uncertainty: npt.ArrayLike | None = None,
        *,
        groups: npt.ArrayLike | None = None,
        y_lower: npt.ArrayLike | None = None,
        y_upper: npt.ArrayLike | None = None,
    ) -> UncertaintyCalibrator:
        """Calibrate on a held-out set.

        Parameters
        ----------
        y_true, y_pred:
            Calibration targets and base-model point predictions.
        uncertainty:
            The DEUP scale ``g(x)`` on the calibration set (``normalized``/``mondrian``).
        groups:
            Per-row group labels (``mondrian`` only).
        y_lower, y_upper:
            Pre-fit quantile predictions on the calibration set (``cqr`` only).
        """
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)

        if self.method == "cqr":
            if y_lower is None or y_upper is None:
                raise ValueError("method='cqr' requires y_lower and y_upper")
            lo = np.asarray(y_lower, dtype=float)
            hi = np.asarray(y_upper, dtype=float)
            scores = np.maximum(lo - yt, yt - hi)
            level = _finite_sample_quantile_level(len(scores), self.alpha)
            self.q_ = float(np.quantile(scores, level, method="higher"))
            self.method_ = "cqr"
            return self

        if uncertainty is None:
            raise ValueError(f"method={self.method!r} requires uncertainty")
        u = np.asarray(uncertainty, dtype=float) + self.eps
        norm_resid = np.abs(yt - yp) / u

        if self.method == "normalized":
            level = _finite_sample_quantile_level(len(norm_resid), self.alpha)
            self.q_ = float(np.quantile(norm_resid, level, method="higher"))
        elif self.method == "mondrian":
            if groups is None:
                raise ValueError("method='mondrian' requires groups")
            g = np.asarray(groups)
            self.group_q_: dict[Any, float] = {}
            for label in np.unique(g):
                mask = g == label
                resid = norm_resid[mask]
                level = _finite_sample_quantile_level(len(resid), self.alpha)
                self.group_q_[label] = float(np.quantile(resid, level, method="higher"))
            # Global fallback for unseen groups at predict time.
            level = _finite_sample_quantile_level(len(norm_resid), self.alpha)
            self.q_ = float(np.quantile(norm_resid, level, method="higher"))
        else:
            raise ValueError(f"Unknown method: {self.method!r}")
        self.method_ = self.method
        return self

    # ------------------------------------------------------------- predict
    def predict_interval(
        self,
        y_pred: npt.ArrayLike,
        uncertainty: npt.ArrayLike | None = None,
        *,
        groups: npt.ArrayLike | None = None,
        y_lower: npt.ArrayLike | None = None,
        y_upper: npt.ArrayLike | None = None,
    ) -> ConformalResult:
        """Return calibrated ``(lower, upper)`` interval bounds for new points."""
        if not hasattr(self, "method_"):
            raise RuntimeError("UncertaintyCalibrator must be fit before predict_interval")

        if self.method_ == "cqr":
            if y_lower is None or y_upper is None:
                raise ValueError("method='cqr' requires y_lower and y_upper")
            lo = np.asarray(y_lower, dtype=float) - self.q_
            hi = np.asarray(y_upper, dtype=float) + self.q_
            return ConformalResult(lower=lo, upper=hi)

        yp = np.asarray(y_pred, dtype=float)
        if uncertainty is None:
            raise ValueError(f"method={self.method_!r} requires uncertainty")
        u = np.asarray(uncertainty, dtype=float) + self.eps

        if self.method_ == "mondrian":
            if groups is None:
                raise ValueError("method='mondrian' requires groups")
            g = np.asarray(groups)
            q = np.array([self.group_q_.get(label, self.q_) for label in g], dtype=float)
        else:
            q = np.full(yp.shape[0], self.q_, dtype=float)

        half = q * u
        return ConformalResult(lower=yp - half, upper=yp + half)


def deup_normalizer(estimator: Any) -> Any:
    """Return a callable ``X -> g(x)`` for use as a MAPIE-style normalization function.

    MAPIE's locally adaptive conformal methods accept a per-point scale; passing a
    fitted DEUP estimator's epistemic estimate as that scale yields DEUP-normalized
    conformal intervals **inside MAPIE**. The returned callable exposes ``predict`` so
    it can stand in as a residual-scaling model.

    Example
    -------
    >>> from deup import DEUPRegressor
    >>> from deup.calibration import deup_normalizer
    >>> model = DEUPRegressor().fit(X_train, y_train)
    >>> normalizer = deup_normalizer(model)
    >>> scale = normalizer.predict(X_cal)   # == model.predict_epistemic(X_cal)
    """

    class _DEUPNormalizer:
        def __init__(self, est: Any) -> None:
            self._est = est

        def fit(self, X: Any, y: Any = None) -> _DEUPNormalizer:  # noqa: ARG002
            return self

        def predict(self, X: Any) -> npt.NDArray[Any]:
            return np.asarray(self._est.predict_epistemic(X), dtype=float)

    return _DEUPNormalizer(estimator)
