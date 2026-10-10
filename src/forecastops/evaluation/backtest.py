import pandas as pd

from forecastops.evaluation.metrics import segmented
from forecastops.features.build import build
from forecastops.training.models import fit, predict


def cutoffs(data, horizon, folds, min_train_days):
    # Last full horizon is untouched by CV model selection.
    holdout_origin = data.date.max() - pd.Timedelta(days=horizon)
    origins = [holdout_origin - pd.Timedelta(days=horizon * i) for i in range(folds, 0, -1)]
    if (origins[0] - data.date.min()).days + 1 < min_train_days:
        raise ValueError("Insufficient history for minimum training + CV + final holdout")
    return origins, holdout_origin


def evaluate(data, name, origins, horizon, seed):
    predictions = []
    for origin in origins:
        history = data[data.date.le(origin)]
        train = build(history, horizon, training=True)
        future = build(history, horizon, training=False)
        model = fit(name, train, seed)
        future["prediction"] = predict(name, model, future)
        truth = data[["store_id", "item_id", "date", "demand"]].rename(
            columns={"date": "target_date", "demand": "target"}
        )
        scored = future.merge(
            truth, on=["store_id", "item_id", "target_date"], validate="one_to_one"
        )
        if len(scored) != len(future):
            raise ValueError("Missing actuals in evaluation window")
        predictions.append(scored)
    result = pd.concat(predictions, ignore_index=True)
    return result, segmented(result)
