import pandas as pd
import pytest

from forecastops.data.m5 import ingest


def test_m5_calendar_price_join_and_subset(tmp_path):
    pd.DataFrame(
        [
            {"store_id": store, "item_id": item, "cat_id": "FOODS", "d_1": 2, "d_2": 5}
            for store in ["CA_1", "TX_1"]
            for item in ["A", "B"]
        ]
    ).to_csv(tmp_path / "sales_train_validation.csv", index=False)
    pd.DataFrame(
        {"d": ["d_1", "d_2"], "date": ["2024-01-01", "2024-01-02"], "wm_yr_wk": [1, 1]}
    ).to_csv(tmp_path / "calendar.csv", index=False)
    pd.DataFrame(
        [
            {"store_id": store, "item_id": "B", "wm_yr_wk": 1, "sell_price": 3.5}
            for store in ["CA_1", "TX_1"]
        ]
    ).to_csv(tmp_path / "sell_prices.csv", index=False)
    result = ingest(tmp_path, tmp_path / "out.csv", ["CA_1", "TX_1"], items=1, days=2)
    assert len(result) == 4
    assert set(result.item_id) == {"B"}  # A is excluded, not filled with invented prices.
    assert result.price.eq(3.5).all()
    assert result.demand.sum() == 14
    with pytest.raises(ValueError, match="Only 1"):
        ingest(tmp_path, tmp_path / "out.csv", ["CA_1", "TX_1"], items=2, days=2)
