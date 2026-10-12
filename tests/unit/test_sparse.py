import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forecastops.features.build import CATEGORICAL, FEATURES, NUMERIC, build
from forecastops.features.sparse import SparseFeatures
from forecastops.training.models import fit, predict


def test_sparse_matches_dense_and_retains_numeric_zeros(sales):
    frame = build(sales, 7, True)
    frame["price_change"] = 0.0
    dense = ColumnTransformer(
        [
            ("cat", OneHotEncoder(sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    ).fit_transform(frame)
    encoded = SparseFeatures().fit_transform(frame)
    assert sparse.isspmatrix_csr(encoded)
    np.testing.assert_allclose(encoded.toarray(), dense)
    assert encoded.nnz == len(frame) * (len(CATEGORICAL) + len(NUMERIC))
    assert (encoded.data == 0).any()
    # Linear regression must retain the dense objective, including its intercept.
    expected = Ridge(alpha=10).fit(dense, frame.target).predict(dense)
    actual = fit("ridge", frame).predict(frame[FEATURES])
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_sparse_tree_roundtrip_and_unknown_categories(sales, tmp_path):
    import joblib

    frame = build(sales, 7, True)
    model = fit("xgboost", frame)
    future = build(sales, 7, False)
    before = predict("xgboost", model, future)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(before, predict("xgboost", joblib.load(path), future))
    future["item_id"] = "unseen"
    assert np.isfinite(predict("xgboost", model, future)).all()
