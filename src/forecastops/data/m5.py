from pathlib import Path

import pandas as pd

from forecastops.data.validate import validate


def ingest(directory: Path, output: Path, stores: list[str], items=100, days=730):
    """Import official, locally downloaded M5 competition CSVs in bounded chunks.

    Keep only series with complete observed prices; pre-launch rows aren't zero-filled.
    Events and future prices deliberately aren't features in V1.
    """
    if items < 1 or days < 1 or not stores:
        raise ValueError("Positive items/days and at least one store are required")
    sales_path = directory / "sales_train_evaluation.csv"
    if not sales_path.exists():
        sales_path = directory / "sales_train_validation.csv"
    calendar = pd.read_csv(directory / "calendar.csv", usecols=["d", "date", "wm_yr_wk"])
    price = pd.read_csv(directory / "sell_prices.csv")
    columns = pd.read_csv(sales_path, nrows=0).columns
    day_cols = [c for c in columns if c.startswith("d_")][-days:]
    chunks = []
    for chunk in pd.read_csv(
        sales_path, chunksize=1000, usecols=["store_id", "item_id", "cat_id", *day_cols]
    ):
        chunks.append(chunk[chunk.store_id.isin(stores)])
    sales = pd.concat(chunks, ignore_index=True)
    if set(stores) - set(sales.store_id):
        raise ValueError("Requested stores not found in M5 files")
    # Require the same selected items in every requested store.
    common = sales.groupby("item_id").store_id.nunique()
    eligible = sorted(common[common.eq(len(set(stores)))].index)
    selected = []
    for item in eligible:
        panel = sales[sales.item_id.eq(item)].melt(
            id_vars=["store_id", "item_id", "cat_id"], var_name="d", value_name="demand"
        )
        panel = panel.merge(calendar, on="d", validate="many_to_one").merge(
            price, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one"
        )
        if panel.sell_price.notna().all():
            selected.append(panel)
        if len(selected) == items:
            break
    if len(selected) < items:
        raise ValueError(f"Only {len(selected)} items have complete prices; reduce --items/--days")
    result = validate(
        pd.concat(selected).rename(columns={"cat_id": "category", "sell_price": "price"})
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
