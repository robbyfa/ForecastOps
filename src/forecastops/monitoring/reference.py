def build_reference(data, predictions):
    """Balanced daily panels contribute the same recent window from every series."""
    sample = data.groupby(["store_id", "item_id"]).tail(28)
    return {
        "demand": sample.demand.tolist(),
        "price": sample.price.tolist(),
        "prediction": predictions.prediction.tolist(),
        "policy": "equal-weight series, last 28 observed days; all winner CV predictions",
        "sampled_series": sample[["store_id", "item_id"]].drop_duplicates().values.tolist(),
        "observations": len(sample),
    }
