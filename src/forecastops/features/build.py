import pandas as pd

from forecastops.data.validate import KEYS

CATEGORICAL = ["store_id", "item_id", "category"]
NUMERIC = [
    "horizon",
    "target_weekday",
    "target_month",
    "target_weekend",
    "price",
    "price_change",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
    "seasonal",
]
FEATURES = CATEGORICAL + NUMERIC


def build(history: pd.DataFrame, horizon: int, training: bool) -> pd.DataFrame:
    """Origin is end-of-day after sales close. lag_1 is demand observed that day.

    Training labels never exceed history's cutoff. Serving needs only historical rows.
    Prices are last observed, not the realized target-day prices.
    """
    if not 1 <= horizon <= 14:
        raise ValueError("Horizon must be between 1 and 14")
    df = history.sort_values([*KEYS, "date"]).copy()
    g = df.groupby(KEYS, sort=False)
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = g.demand.shift(lag - 1)
    for window in [7, 28]:
        df[f"rolling_mean_{window}"] = g.demand.transform(lambda s, w=window: s.rolling(w).mean())
    df["rolling_std_28"] = g.demand.transform(lambda s: s.rolling(28).std(ddof=0))
    df["price_change"] = g.price.pct_change(fill_method=None)
    parts = []
    for h in range(1, horizon + 1):
        part = df.copy()
        part["origin"] = part.date
        part["target_date"] = part.date + pd.Timedelta(days=h)
        part["horizon"] = h
        part["target_weekday"] = part.target_date.dt.dayofweek
        part["target_month"] = part.target_date.dt.month
        part["target_weekend"] = part.target_weekday.ge(5).astype(int)
        part["seasonal"] = g.demand.shift(6 - (h - 1) % 7)
        if training:
            part["target"] = g.demand.shift(-h)
        else:
            part = part[part.date.eq(history.date.max())]
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    required = FEATURES + (["target"] if training else [])
    return result.dropna(subset=required).reset_index(drop=True)
