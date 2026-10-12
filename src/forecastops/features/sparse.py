"""CSR features with explicit numeric zeros for XGBoost's missing-value semantics."""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forecastops.features.build import CATEGORICAL, NUMERIC


class SparseFeatures(TransformerMixin, BaseEstimator):
    def fit(self, frame, y=None):
        self.numeric_ = list(NUMERIC)
        self.categorical_ = list(CATEGORICAL)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.encoder_.fit(frame[self.categorical_])
        self.scaler_ = StandardScaler().fit(frame[self.numeric_])
        return self

    def transform(self, frame):
        categorical = self.encoder_.transform(frame[self.categorical_])
        values = self.scaler_.transform(frame[self.numeric_])
        rows, columns = values.shape
        # csr_matrix(dense) would discard numeric zeros, turning them into missing
        # values in tree boosters. Retain all numeric entries, including zero.
        numeric = sparse.csr_matrix(
            (
                values.ravel(),
                np.tile(np.arange(columns, dtype=np.int32), rows),
                np.arange(rows + 1, dtype=np.int32) * columns,
            ),
            shape=values.shape,
        )
        return sparse.hstack([categorical, numeric], format="csr")
