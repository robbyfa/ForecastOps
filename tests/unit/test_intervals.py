import numpy as np
import pandas as pd
import pytest

from forecastops.evaluation.intervals import bounds, calibrate, coverage


def test_horizon_order_statistic_clipping_and_coverage():
    frame = pd.DataFrame(
        {
            "horizon": [1] * 19 + [2] * 19,
            "target": list(range(19)) + list(range(0, 38, 2)),
            "prediction": [0] * 38,
        }
    )
    interval = calibrate(frame, 2)
    assert interval["by_horizon"]["1"] == {"n": 19, "rank": 18, "radius": 17.0}
    assert interval["by_horizon"]["2"]["radius"] == 34.0
    lower, upper = bounds([2, 2], [1, 2], interval)
    np.testing.assert_array_equal(lower, [0, 0])
    np.testing.assert_array_equal(upper, [19, 36])
    measured, _ = coverage(frame, interval)
    assert measured["overall"]["coverage"] == pytest.approx(18 / 19)
    with pytest.raises(ValueError, match="Insufficient"):
        calibrate(frame.iloc[:2], 2)


def test_holdout_cannot_change_calibration_or_fitted_model(sales, settings, tmp_path):
    import joblib

    from forecastops.features.build import build
    from forecastops.training.models import predict
    from forecastops.training.pipeline import train

    first = train(sales, settings, tracking=False)
    cutoff = pd.Timestamp(first["holdout_origin"])
    changed = sales.copy()
    changed.loc[changed.date.gt(cutoff), "demand"] = 99999
    settings.artifact_dir = tmp_path / "separate-run"
    second = train(changed, settings, tracking=False)
    assert first["name"] == second["name"]
    assert first["interval"] == second["interval"]
    assert max(pd.to_datetime(first["cv_origins"])) + pd.Timedelta(
        days=settings.horizon
    ) <= pd.Timestamp(first["train_end"])
    assert pd.Timestamp(first["train_end"]) == pd.Timestamp(first["calibration_origins"][0])
    assert (
        pd.Timestamp(first["calibration_origins"][-1]) + pd.Timedelta(days=settings.horizon)
        == cutoff
    )
    inputs = build(sales[sales.date.le(cutoff)], settings.horizon, False)
    a, b = [joblib.load(r["model_path"]) for r in (first, second)]
    np.testing.assert_array_equal(
        predict(a["name"], a["model"], inputs), predict(b["name"], b["model"], inputs)
    )
