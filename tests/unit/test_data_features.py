import numpy as np
import pandas as pd
import pytest

from forecastops.data.validate import validate
from forecastops.features.build import FEATURES, build


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "gap",
        "negative",
        "zero_price",
        "null",
        "infinite",
        "category",
        "empty_key",
        "intraday",
    ],
)
def test_invalid_data_rejected(sales, damage):
    bad = sales.copy()
    if damage == "duplicate":
        bad = pd.concat([bad, bad.iloc[:1]])
    elif damage == "gap":
        bad = bad.drop(index=20)
    elif damage == "negative":
        bad.loc[0, "demand"] = -1
    elif damage == "zero_price":
        bad.loc[0, "price"] = 0
    elif damage == "null":
        bad.loc[0, "item_id"] = None
    elif damage == "infinite":
        bad["demand"] = bad.demand.astype(float)
        bad.loc[0, "demand"] = np.inf
    elif damage == "category":
        bad.loc[0, "category"] = "OTHER"
    elif damage == "empty_key":
        bad.loc[0, "item_id"] = ""
    else:
        bad.loc[0, "date"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        validate(bad)


def test_training_features_do_not_change_with_future_actuals(sales):
    origin = sales.date.min() + pd.Timedelta(days=90)
    original = build(sales, 14, True)
    changed = sales.copy()
    changed.loc[changed.date.gt(origin), ["demand", "price"]] = 99999
    altered = build(changed, 14, True)
    pd.testing.assert_frame_equal(
        original.loc[original.origin.eq(origin), FEATURES],
        altered.loc[altered.origin.eq(origin), FEATURES],
    )


def test_training_labels_and_forecast_inputs_agree(sales):
    origin = sales.date.min() + pd.Timedelta(days=90)
    train = build(sales, 14, True)
    expected = train.loc[train.origin.eq(origin), FEATURES].reset_index(drop=True)
    actual = build(sales[sales.date.le(origin)], 14, False)[FEATURES]
    pd.testing.assert_frame_equal(expected, actual)


def test_baseline_repeats_last_observed_week(sales):
    panel = sales[(sales.store_id == "STORE_1") & (sales.item_id == "ITEM_001")].copy()
    panel["demand"] = np.arange(len(panel))
    future = build(panel, 14, False)
    assert future.seasonal.tolist() == list(range(133, 140)) * 2
    assert future.lag_1.eq(139).all()
    assert future.lag_28.eq(112).all()


def test_training_labels_never_exceed_cutoff(sales):
    frame = build(sales, 14, True)
    assert frame.target_date.max() == sales.date.max()
    assert frame.target_date.gt(frame.origin).all()
