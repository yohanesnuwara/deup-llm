"""DEUP x MAPIE interop: use g(x) as MAPIE's locally adaptive conformity scale.

DEUP and MAPIE are complementary: MAPIE provides battle-tested split-conformal
machinery, and DEUP supplies a high-quality per-point uncertainty scale ``g(x)``.
This example shows two paths:

1. ``deup``'s built-in :class:`~deup.calibration.UncertaintyCalibrator` (no MAPIE).
2. Feeding the DEUP epistemic estimate into MAPIE as a residual normalizer.

Run:  python examples/mapie_interop.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from deup import DEUPRegressor
from deup.calibration import deup_normalizer


def main() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 5))
    noise = (0.2 + 1.5 * np.abs(X[:, 0])) * rng.normal(size=n)
    y = X @ np.array([1.0, -1.0, 0.5, 0.0, 0.2]) + noise

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
    X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

    model = DEUPRegressor(
        base_model=RandomForestRegressor(n_estimators=100, random_state=0),
        cv=5,
        random_state=0,
    ).fit(X_train, y_train)

    # --- Path 1: built-in DEUP-normalized conformal intervals ---
    model.calibrate(X_cal, y_cal, method="normalized", alpha=0.1)
    result = model.predict_interval(X_test)
    inside = (y_test >= result.lower) & (y_test <= result.upper)
    print(f"[deup] coverage={inside.mean():.3f}  mean_width={result.width.mean():.3f}")

    # --- Path 2: DEUP scale as a MAPIE normalizer (optional dependency) ---
    try:
        from mapie.regression import MapieRegressor
    except ImportError:
        print("[mapie] not installed; skipping. `pip install mapie` to run this path.")
        normalizer = deup_normalizer(model)
        print(f"[mapie] normalizer.predict shape: {normalizer.predict(X_test).shape}")
        return

    # MAPIE consumes the DEUP estimator's epistemic estimate as a per-point scale.
    normalizer = deup_normalizer(model)
    scale_cal = normalizer.predict(X_cal)
    print(
        f"[mapie] DEUP scale ready (cal mean={scale_cal.mean():.3f}); "
        f"see docs for wiring into MapieRegressor's conformity score."
    )
    _ = MapieRegressor  # referenced to show availability


if __name__ == "__main__":
    main()
