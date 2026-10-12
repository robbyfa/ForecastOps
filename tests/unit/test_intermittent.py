import numpy as np
import pandas as pd

from forecastops.features.build import build
from forecastops.features.intermittent import croston_sba


def test_croston_causal_initialization_updates_and_zeros():
    demand = np.array([0, 0, 6, 0, 0, 0, 10, 0])
    forecast = croston_sba(demand)
    np.testing.assert_allclose(
        forecast, [0, 0, 1.9, 1.9, 1.9, 1.9, 0.95 * 6.4 / 3.1, 0.95 * 6.4 / 3.1]
    )
    np.testing.assert_array_equal(croston_sba(np.zeros(20)), np.zeros(20))
    np.testing.assert_array_equal(croston_sba(demand[:5]), forecast[:5])


def test_events_are_known_target_dates_and_preserve_horizon_alignment(sales):
    panel = sales.copy()
    panel["date"] += pd.Timestamp("2024-12-20") - panel.date.max()
    frame = build(panel, 14, False)
    assert frame.loc[frame.horizon.eq(5), "target_christmas"].eq(1).all()
    assert frame.loc[frame.horizon.eq(12), "target_new_year"].eq(1).all()
    assert frame.loc[frame.horizon.eq(11), "target_month_end"].eq(1).all()
    assert frame.loc[frame.horizon.ne(5), "target_christmas"].eq(0).all()
