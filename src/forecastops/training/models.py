import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from forecastops.features.build import CATEGORICAL, FEATURES, NUMERIC

NAMES = ["seasonal_naive", "moving_average", "ridge", "xgboost"]
PARAMETERS = {
    "seasonal_naive": {"season_length": 7},
    "moving_average": {"window": 7},
    "ridge": {"alpha": 10},
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
    if name in ["seasonal_naive", "moving_average"]:
        return None
    preprocess = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("numeric", StandardScaler(), NUMERIC),
        ]
    )
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
    else:
        result = model.predict(frame[FEATURES])
    result = np.maximum(0, result)
    if not np.isfinite(result).all():
        raise ValueError("Model emitted nonfinite predictions")
    return result
