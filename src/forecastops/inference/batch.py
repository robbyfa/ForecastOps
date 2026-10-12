import joblib
import numpy as np
import pandas as pd

from forecastops.data.validate import validate
from forecastops.evaluation.intervals import bounds
from forecastops.features.build import build
from forecastops.features.serving import model_input
from forecastops.registry.store import current
from forecastops.training.models import predict


def forecast(data, settings, database):
    champion = current(settings)
    if champion is None:
        raise ValueError("No promoted champion. Inspect latest-training.json quality gates")
    data = validate(data)
    if data.date.max() < pd.Timestamp(champion.get("data_end", champion["train_end"])):
        raise ValueError("Cannot forecast before champion data cutoff")
    expected = {tuple(k) for k in champion["series_keys"]}
    actual = set(map(tuple, data[["store_id", "item_id", "category"]].drop_duplicates().values))
    if actual != expected:
        raise ValueError("Series/category universe differs from champion; review and retrain")
    future = build(data, champion["horizon"], training=False)
    if len(future) != len(expected) * champion["horizon"]:
        raise ValueError("Insufficient history: require at least 28 complete days per series")
    if "model_uri" in champion:
        import mlflow

        mlflow.set_tracking_uri(champion["tracking_uri"])
        prediction = mlflow.pyfunc.load_model(champion["model_uri"]).predict(
            model_input(future, champion.get("features"), legacy="interval" not in champion)
        )
    else:
        bundle = joblib.load(champion["model_path"])
        prediction = predict(champion["name"], bundle["model"], future)
    if not np.isfinite(prediction).all() or (np.asarray(prediction) < 0).any():
        raise ValueError("Invalid forecast values")
    future["prediction"] = prediction
    if "interval" in champion:
        future["lower_bound"], future["upper_bound"] = bounds(
            prediction, future.horizon, champion["interval"]
        )
    else:  # Existing immutable artifacts retain their original scalar intervals.
        future["lower_bound"] = np.maximum(0, prediction - champion["interval_radius"])
        future["upper_bound"] = prediction + champion["interval_radius"]
    future["model_version"] = champion["version"]
    future["forecast_date"] = future.origin.dt.date
    future["target_date"] = future.target_date.dt.date
    columns = [
        "forecast_date",
        "target_date",
        "store_id",
        "item_id",
        "category",
        "prediction",
        "lower_bound",
        "upper_bound",
        "model_version",
    ]
    rows = future[columns].to_dict("records")
    database.initialize()
    batch_id = database.save(rows)
    for row in rows:
        row["batch_id"] = batch_id
    return rows
