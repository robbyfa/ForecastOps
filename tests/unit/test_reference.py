import pandas as pd

from forecastops.monitoring.reference import build_reference


def test_long_sorted_histories_do_not_exclude_early_series():
    data = pd.DataFrame(
        {
            "store_id": ["A"] * 3000 + ["B"] * 3000 + ["C"] * 3000,
            "item_id": "item",
            "demand": [1] * 3000 + [10] * 3000 + [100] * 3000,
            "price": 5,
        }
    )
    result = build_reference(data, pd.DataFrame({"prediction": [3, 4]}))
    assert result["sampled_series"] == [["A", "item"], ["B", "item"], ["C", "item"]]
    assert pd.Series(result["demand"]).value_counts().to_dict() == {1: 28, 10: 28, 100: 28}
    assert result["prediction"] == [3, 4]
