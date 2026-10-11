from argparse import Namespace
from unittest.mock import Mock

import pytest

from forecastops.cli import execute
from forecastops.registry.store import write_json


@pytest.mark.parametrize("days,expected", [(3, "deferred"), (20, "failed")])
def test_cycle_keeps_publishing_when_retraining_is_deferred_or_fails(
    sales, settings, tmp_path, monkeypatch, days, expected
):
    from datetime import timedelta

    path = tmp_path / "sales.csv"
    sales.to_csv(path, index=False)
    end = sales.date.max().date()
    write_json(
        settings.artifact_dir / "champion.json",
        {
            "version": "old",
            "train_end": str(end - timedelta(days=days)),
        },
    )
    train = Mock(side_effect=RuntimeError("training failure"))
    forecast = Mock(return_value=[{"model_version": "old", "forecast_date": end}])
    monkeypatch.setattr("forecastops.training.pipeline.train", train)
    monkeypatch.setattr("forecastops.inference.batch.forecast", forecast)
    monkeypatch.setattr(
        "forecastops.monitoring.report.monitor",
        Mock(
            return_value={
                "retraining_required": True,
                "model_version": "old",
                "as_of": str(end),
            }
        ),
    )
    result = execute(Namespace(command="cycle", data=path, no_tracking=True), settings)
    assert result["retraining"] == expected
    assert result["model_version"] == "old"
    forecast.assert_called_once()
    assert train.call_count == (1 if expected == "failed" else 0)
    assert not (settings.artifact_dir / "last-retrain-request.txt").exists()
