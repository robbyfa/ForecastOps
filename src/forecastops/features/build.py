import pandas as pd

from forecastops.data.validate import KEYS
from forecastops.features.intermittent import croston_sba

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
    "croston_sba",
    "target_month_end",
    "target_new_year",
    "target_christmas",
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
    df["croston_sba"] = df.groupby(KEYS, sort=False).demand.transform(croston_sba)
    if not training:
        df = df.groupby(KEYS, sort=False).tail(28).copy()
    for column in CATEGORICAL:
        df[column] = df[column].astype("category")
    g = df.groupby(KEYS, sort=False, observed=True)
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = g.demand.shift(lag - 1)
    for window in [7, 28]:
        df[f"rolling_mean_{window}"] = g.demand.transform(lambda s, w=window: s.rolling(w).mean())
    df["rolling_std_28"] = g.demand.transform(lambda s: s.rolling(28).std(ddof=0))
    df["price_change"] = g.price.pct_change(fill_method=None)
    parts = []
    required = FEATURES + (["target"] if training else [])
    for h in range(1, horizon + 1):
        selected = df.index if training else df.index[df.date.eq(history.date.max())]
        part = df.loc[selected].copy()
        part["origin"] = part.date
        part["target_date"] = part.date + pd.Timedelta(days=h)
        part["horizon"] = h
        part["target_weekday"] = part.target_date.dt.dayofweek
        part["target_month"] = part.target_date.dt.month
        part["target_weekend"] = part.target_weekday.ge(5).astype(int)
        # Deterministic dates are known at every origin. No realized future
        # prices, promotion outcomes or retrospectively supplied event labels.
        part["target_month_end"] = part.target_date.dt.is_month_end.astype(int)
        part["target_new_year"] = (
            (part.target_month == 1) & (part.target_date.dt.day == 1)
        ).astype(int)
        part["target_christmas"] = (
            (part.target_month == 12) & (part.target_date.dt.day == 25)
        ).astype(int)
        part["seasonal"] = g.demand.shift(6 - (h - 1) % 7)
        if training:
            part["target"] = g.demand.shift(-h)
        parts.append(part.dropna(subset=required))
    result = pd.concat(parts, ignore_index=True)
    return result
