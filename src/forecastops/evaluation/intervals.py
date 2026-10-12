"""Chronological, horizon-specific residual calibration for a frozen predictor.

The split-conformal order statistic is used, but dependent retail series and
time drift do not satisfy an exchangeability guarantee. Report observed coverage.
"""

import math

import numpy as np


def calibrate(frame, horizon, coverage=0.9):
    result = {}
    for h in range(1, horizon + 1):
        group = frame.loc[frame.horizon.eq(h)]
        scores = np.abs(group.target.to_numpy() - group.prediction.to_numpy())
        n = len(scores)
        rank = math.ceil((n + 1) * coverage)
        if not n or rank > n or not np.isfinite(scores).all():
            raise ValueError(
                f"Insufficient finite calibration scores at horizon {h}; "
                "increase calibration_windows"
            )
        result[str(h)] = {
            "radius": float(np.partition(scores, rank - 1)[rank - 1]),
            "n": n,
            "rank": rank,
        }
    return {
        "target_coverage": coverage,
        "by_horizon": result,
        "method": "chronological split residuals; ceil((n+1)*coverage) order statistic",
        "limitation": "Dependent series/time drift: nominal coverage is not guaranteed",
    }


def bounds(prediction, horizons, interval):
    radii = np.array([interval["by_horizon"][str(int(h))]["radius"] for h in horizons])
    if not np.isfinite(radii).all() or (radii < 0).any():
        raise ValueError("Invalid calibrated interval radii")
    return np.maximum(0, np.asarray(prediction) - radii), np.asarray(prediction) + radii


def coverage(frame, interval):
    lower, upper = bounds(frame.prediction, frame.horizon, interval)
    scored = frame.assign(lower_bound=lower, upper_bound=upper)

    def measure(group):
        return {
            "n": len(group),
            "coverage": float(group.target.between(group.lower_bound, group.upper_bound).mean()),
            "mean_width": float((group.upper_bound - group.lower_bound).mean()),
        }

    return {
        "overall": measure(scored),
        "by_horizon": {str(h): measure(g) for h, g in scored.groupby("horizon")},
        "by_store": {str(k): measure(g) for k, g in scored.groupby("store_id", observed=True)}
        if "store_id" in scored
        else {},
        "by_category": {str(k): measure(g) for k, g in scored.groupby("category", observed=True)}
        if "category" in scored
        else {},
    }, scored
