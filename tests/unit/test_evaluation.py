import pytest

from forecastops.evaluation.backtest import cutoffs
from forecastops.evaluation.metrics import metrics
from forecastops.registry.gates import quality_gate


def test_zero_demand_metrics_are_explicit():
    assert metrics([0, 0], [0, 0])["wape"] is None
    assert metrics([0, 0], [0, 0])["smape"] == 0
    assert metrics([1, 3], [2, 2])["wape"] == 0.5
    with pytest.raises(ValueError):
        metrics([], [])


def score(wape, segment=1):
    return {"overall": {"wape": wape}, "store_id:A": {"mae": segment}}


def test_gate_rejects_segment_regression_and_slow_prediction():
    result = quality_gate(score(0.1, 3), score(0.2), None, 0.02, 0.1, 31, 30)
    assert not result["passed"]
    assert len(result["reasons"]) == 2


def test_gate_compares_incumbent():
    assert not quality_gate(score(0.1), score(0.2), score(0.09), 0.02, 0.1, 1, 30)["passed"]
    assert quality_gate(score(0.1), score(0.2), score(0.15), 0.02, 0.1, 1, 30)["passed"]


def test_holdout_is_separate(sales):
    origins, holdout = cutoffs(sales, 14, 3, 70)
    assert (holdout - origins[-1]).days == 14
    assert (sales.date.max() - holdout).days == 14
    with pytest.raises(ValueError):
        cutoffs(sales, 14, 10, 70)
