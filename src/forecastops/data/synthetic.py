from pathlib import Path

import numpy as np
import pandas as pd


def generate(path: Path, days=240, stores=3, items=8, seed=42):
    """Deterministic simulated retail data; never represented as M5 measurements."""
    if days < 1 or stores < 1 or items < 1:
        raise ValueError("days, stores and items must be positive")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days)
    rows = []
    for s in range(stores):
        for i in range(items):
            base = 8 + i * 3 + s * 2
            price = 2.5 + i * 0.7
            for t, date in enumerate(dates):
                weekly = [0.7, 0.8, 0.85, 1, 1.2, 1.5, 1.4][date.dayofweek]
                mean = base * weekly * (1 + 0.0015 * t)
                demand = max(0, round(mean + rng.normal(0, 1.5)))
                rows.append(
                    (
                        f"STORE_{s + 1}",
                        f"ITEM_{i + 1:03}",
                        ["FOODS", "HOBBIES", "HOUSEHOLD"][i % 3],
                        date,
                        demand,
                        price,
                    )
                )
    df = pd.DataFrame(rows, columns=["store_id", "item_id", "category", "date", "demand", "price"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
