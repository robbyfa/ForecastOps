from datetime import timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from forecastops.api.app import create_app
from forecastops.inference.batch import forecast
from forecastops.inference.database import Database
from forecastops.monitoring.report import monitor, psi
from forecastops.registry.store import current, write_json
from forecastops.training.pipeline import train


@pytest.fixture
def trained(sales, settings):
    report = train(sales, settings, tracking=False)
    return sales, settings, report


def test_training_audit_and_gate(trained):
    sales, settings, report = trained
    assert set(report["models"]) == {"seasonal_naive", "moving_average", "ridge", "xgboost"}
    assert len(report["dataset_sha256"]) == 64
    assert (settings.artifact_dir / "latest-training.json").exists()
    assert bool(current(settings)) == report["quality_gate"]["passed"]


def publish_for_test(settings, report):
    # Serving contract tests are independent of whether this small fixture passes the gate.
    write_json(settings.artifact_dir / "champion.json", report)


def test_forecast_api_and_idempotent_storage(trained):
    sales, settings, report = trained
    publish_for_test(settings, report)
    db = Database(settings.database_url)
    rows = forecast(sales, settings, db)
    assert len(rows) == 2 * 3 * 7
    forecast(sales, settings, db)
    assert len(db.query()) == len(rows)
    assert all(r["lower_bound"] <= r["prediction"] <= r["upper_bound"] for r in rows)
    with TestClient(
        create_app(settings, db, clock=lambda: sales.date.max().tz_localize("UTC").to_pydatetime())
    ) as client:
        response = client.get("/forecasts/items/ITEM_001?store_id=STORE_1&horizon=7")
        assert response.status_code == 200
        assert len(response.json()) == 7
        assert response.json()[0]["model_version"] == report["version"]
        assert client.get("/forecasts/items/ITEM_001?horizon=15").status_code == 422
        assert client.get("/forecasts/items/MISSING").status_code == 404
        assert client.get("/health").status_code == 200
        assert client.get("/models/current").json()["version"] == report["version"]
        assert "forecastops_requests_total" in client.get("/metrics").text
        assert len(client.get("/forecasts/stores/STORE_1?limit=2&offset=2").json()) == 2


def test_one_batch_does_not_meet_per_horizon_monitoring_coverage(trained):
    sales, settings, report = trained
    publish_for_test(settings, report)
    db = Database(settings.database_url)
    rows = forecast(sales, settings, db)
    waiting = monitor(sales, settings, db, replay=True)
    assert waiting["status"] == "awaiting_actuals"
    assert not waiting["retraining_required"]
    new = pd.DataFrame(rows).rename(columns={"target_date": "date"})
    new["demand"], new["price"] = 1000, 5
    actuals = pd.concat([sales, new[sales.columns]], ignore_index=True)
    actuals["date"] = pd.to_datetime(actuals.date)
    degraded = monitor(actuals, settings, db, replay=True)
    assert degraded["coverage"] == pytest.approx(1 / settings.horizon)
    assert not degraded["retraining_required"]
    assert degraded["status"] == "awaiting_actuals"
    assert degraded["drift_alert"]
    assert not (settings.artifact_dir / "retrain-request.json").exists()


def test_no_champion_or_time_travel(sales, settings):
    db = Database(settings.database_url)
    with pytest.raises(ValueError, match="No promoted"):
        forecast(sales, settings, db)
    with TestClient(create_app(settings, db)) as client:
        assert client.get("/models/current").status_code == 404
        assert client.get("/forecasts/items/ITEM_001").status_code == 404


def test_failed_retraining_preserves_champion(trained):
    sales, settings, report = trained
    publish_for_test(settings, report)
    settings.min_improvement = 0.99
    result = train(sales, settings, tracking=False)
    assert not result["quality_gate"]["passed"]
    assert current(settings)["version"] == report["version"]


def test_frozen_incumbent_uses_only_post_training_actuals(trained):
    from forecastops.data.synthetic import generate
    from forecastops.data.validate import validate

    sales, settings, report = trained
    publish_for_test(settings, report)
    # Extend each series, retaining the exact original historical observations.
    longer = validate(generate(settings.artifact_dir / "longer.csv", 154, 2, 3))
    future = longer[longer.date.gt(sales.date.max())]
    extended = pd.concat([sales, future], ignore_index=True)
    result = train(extended, settings, tracking=False)
    assert "incumbent" in result["holdout"]
    assert result["holdout_origin"] >= report["train_end"]


def test_constant_reference_drift_detected():
    assert psi([5] * 100, [5] * 100) == 0
    assert psi([5] * 100, [10] * 100) > 0.2


def test_batch_transaction_rolls_back_on_invalid_row(settings):
    from datetime import date

    db = Database(settings.database_url)
    db.initialize()
    origin = date(2024, 1, 1)
    row = dict(
        forecast_date=origin,
        target_date=origin + timedelta(days=1),
        store_id="A",
        item_id="B",
        category="C",
        prediction=3,
        lower_bound=1,
        upper_bound=5,
        model_version="v1",
    )
    db.save([row])
    with pytest.raises(IntegrityError):
        db.save([{**row, "prediction": None}])
    assert db.query()[0]["prediction"] == 3
