import numpy as np


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Metrics require nonempty matching arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    volume = np.abs(actual).sum()
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape": float(np.abs(error).sum() / volume) if volume > 0 else None,
        "smape": float(
            np.divide(
                2 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0
            ).mean()
        ),
        "bias": float(error.mean()),
    }


def segmented(frame):
    result = {"overall": metrics(frame.target, frame.prediction)}
    for dimension in ["store_id", "category"]:
        for name, group in frame.groupby(dimension, observed=True):
            result[f"{dimension}:{name}"] = metrics(group.target, group.prediction)
    return result
