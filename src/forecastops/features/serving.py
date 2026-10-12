"""Portable MLflow wire types; pandas categoricals stay inside feature construction."""

from forecastops.features.build import CATEGORICAL, FEATURES


def model_input(frame, features=None, *, legacy=False):
    result = frame[features or FEATURES].copy()
    for column in CATEGORICAL:
        result[column] = result[column].astype(object)
    if not legacy:
        for column in result.columns.difference(CATEGORICAL):
            result[column] = result[column].astype(float)
    return result
