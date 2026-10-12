import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from forecastops.features.build import FEATURES
from forecastops.features.intermittent import ALPHA
from forecastops.features.sparse import SparseFeatures

BASELINES = {"seasonal_naive", "moving_average", "croston_sba"}
NAMES = ["seasonal_naive", "moving_average", "croston_sba", "ridge", "xgboost"]
PARAMETERS = {
    "seasonal_naive": {"season_length": 7},
    "moving_average": {"window": 7},
    "croston_sba": {"alpha": ALPHA, "initialization": "first positive size and elapsed interval"},
    "ridge": {"alpha": 10, "solver": "lsqr", "tol": 1e-8},
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.07,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": 1,
    },
}


def fit(name, frame, seed=42):
    if name in BASELINES:
        return None
    preprocess = SparseFeatures()
    if name == "ridge":
        estimator = Ridge(**PARAMETERS[name])
    elif name == "xgboost":
        estimator = XGBRegressor(**PARAMETERS[name], random_state=seed)
    else:
        raise ValueError(f"Unknown model: {name}")
    return make_pipeline(preprocess, estimator).fit(frame[FEATURES], frame.target)


def predict(name, model, frame):
    if name == "seasonal_naive":
        result = frame.seasonal.to_numpy()
    elif name == "moving_average":
        result = frame.rolling_mean_7.to_numpy()
    elif name == "croston_sba":
        result = frame.croston_sba.to_numpy()
    else:
        result = model.predict(frame[FEATURES])
    result = np.maximum(0, result)
    if not np.isfinite(result).all():
        raise ValueError("Model emitted nonfinite predictions")
    return result
