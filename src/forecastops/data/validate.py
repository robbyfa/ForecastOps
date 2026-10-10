import numpy as np
import pandas as pd

KEYS = ["store_id", "item_id"]
REQUIRED = [*KEYS, "category", "date", "demand", "price"]


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Strict balanced-panel contract. Never silently fill gaps or missing observations."""
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = frame[REQUIRED].copy()
    if df.empty or df.isna().any().any():
        raise ValueError("Dataset is empty or contains nulls")
    for col in [*KEYS, "category"]:
        if not df[col].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"{col} must contain nonempty strings")
    df["date"] = pd.to_datetime(df.date, errors="raise")
    if df.date.dt.tz is not None or not df.date.eq(df.date.dt.normalize()).all():
        raise ValueError("Dates must be timezone-naive calendar dates")
    for col in ["demand", "price"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"{col} must be finite")
    if (df.demand < 0).any() or (df.price <= 0).any():
        raise ValueError("Demand must be nonnegative and price strictly positive")
    if df.duplicated([*KEYS, "date"]).any():
        raise ValueError("Duplicate store/item/date observations")
    if df.groupby(KEYS).category.nunique().gt(1).any():
        raise ValueError("Category changed within a series")
    expected_days = (df.date.max() - df.date.min()).days + 1
    groups = df.groupby(KEYS).date.agg(["min", "max", "size"])
    if not (groups["size"].eq(expected_days)).all():
        raise ValueError("Date gaps or misaligned series: require a common daily date range")
    return df.sort_values([*KEYS, "date"]).reset_index(drop=True)


def read_sales(path) -> pd.DataFrame:
    return validate(pd.read_csv(path, dtype={c: str for c in [*KEYS, "category"]}))
