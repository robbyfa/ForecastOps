from datetime import timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from forecastops.api.app import create_app
from forecastops.inference.database import Database, forecasts
from forecastops.monitoring.report import monitor
from forecastops.registry.store import write_json


@pytest.fixture
def daily_forecasts(sales, settings):
    data = sales.copy()
    data["demand"], data["price"] = 10.0, 5.0
    settings.horizon = 14
    champion = {
        "version": "tested",
        "name": "ridge",
        "horizon": 14,
        "reference": {"demand": [10.0] * 50, "price": [5.0] * 50, "prediction": [10.0] * 50},
        "holdout_by_horizon": {str(h): {"overall": {"wape": 0.1}} for h in range(1, 15)},
        "series_keys": data[["store_id", "item_id", "category"]].drop_duplicates().values.tolist(),
    }
    write_json(settings.artifact_dir / "champion.json", champion)
    end = data.date.max()
    db = Database(settings.database_url)
    db.initialize()
    batches = {}
    for day in pd.date_range(end - pd.Timedelta(days=6), end):
        for horizon in range(1, 15):
            origin = (day - pd.Timedelta(days=horizon)).date()
            for store, item, category in champion["series_keys"]:
                batches.setdefault(origin, []).append(
                    {
                        "forecast_date": origin,
                        "target_date": day.date(),
                        "store_id": store,
                        "item_id": item,
                        "category": category,
                        "prediction": 100 if horizon == 14 else 10,
                        "lower_bound": 0,
                        "upper_bound": 110,
                        "model_version": "tested",
                    }
                )
    for origin, rows in batches.items():
        db.save(rows)
        # Simulate timely issuance; live scoring must reject retrospective backfills.
        with db.engine.begin() as conn:
            conn.execute(
                update(forecasts)
                .where(forecasts.c.forecast_date == origin)
                .values(created_at=pd.Timestamp(origin).tz_localize("UTC").to_pydatetime())
            )
    now = (end.tz_localize("UTC") + pd.Timedelta(days=1)).to_pydatetime()
    return data, settings, db, champion, now


def test_good_horizon_one_does_not_hide_bad_horizon_fourteen(daily_forecasts):
    data, settings, db, _, now = daily_forecasts
    result = monitor(data, settings, db, now=now)
    assert result["by_horizon"]["1"]["performance"]["overall"]["wape"] == 0
    assert result["by_horizon"]["14"]["performance"]["overall"]["wape"] == 9
    assert result["coverage"] == 1
    assert result["degraded_horizons"] == [14]
    assert result["retraining_required"]
    assert (settings.artifact_dir / "retrain-request.json").exists()


def test_legacy_reference_cannot_trigger_retraining(daily_forecasts):
    data, settings, db, champion, now = daily_forecasts
    del champion["holdout_by_horizon"]
    write_json(settings.artifact_dir / "champion.json", champion)
    report = monitor(data, settings, db, now=now)
    assert report["status"] == "reference_unavailable"
    assert not report["retraining_required"]


@pytest.mark.parametrize("replay,days", [(True, 0), (False, 10), (False, -2)])
def test_stale_future_or_replayed_actuals_never_request_training(daily_forecasts, replay, days):
    data, settings, db, _, now = daily_forecasts
    report = monitor(data, settings, db, now=now + timedelta(days=days), replay=replay)
    assert not report["retraining_required"]
    assert not (settings.artifact_dir / "retrain-request.json").exists()
    if not replay:
        assert report["status"] == "stale_data"


def test_late_backfills_are_not_live_issued_forecasts(daily_forecasts):
    data, settings, db, _, now = daily_forecasts
    with db.engine.begin() as conn:
        conn.execute(update(forecasts).values(created_at=now))
    live = monitor(data, settings, db, now=now)
    assert live["matched_actuals"] == 0
    assert not live["retraining_required"]
    replay = monitor(data, settings, db, now=now, replay=True)
    assert replay["would_retrain"]
    assert not replay["retraining_required"]


def test_metrics_expire_and_historical_forecasts_require_explicit_date(daily_forecasts):
    data, settings, db, _, now = daily_forecasts
    monitor(data, settings, db, now=now)
    clock = [now]
    with TestClient(create_app(settings, db, clock=lambda: clock[0])) as client:
        text = client.get("/metrics").text
        assert 'forecastops_horizon_wape{horizon="14"} 9.0' in text
        assert "forecastops_monitoring_fresh 1.0" in text
        assert client.get("/ready").status_code == 200
        clock[0] += timedelta(days=3)
        text = client.get("/metrics").text
        assert "forecastops_monitoring_fresh 0.0" in text
        assert "forecastops_production_wape NaN" in text
        assert "forecastops_retraining_required 0.0" in text
        assert client.get("/ready").status_code == 503
        route = "/forecasts/items/ITEM_001"
        assert client.get(route).status_code == 503
        origin = str((data.date.max() - pd.Timedelta(days=1)).date())
        assert client.get(route, params={"forecast_date": origin}).status_code == 200
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize("kind", ["legacy", "replay", "stale_actuals", "future_report"])
def test_metrics_fail_closed_for_non_live_reports(daily_forecasts, kind):
    data, settings, db, _, now = daily_forecasts
    report = monitor(data, settings, db, now=now)
    if kind == "legacy":
        del report["generated_at"]
    elif kind == "replay":
        report["mode"] = "replay"
    elif kind == "stale_actuals":
        report["as_of"] = "2020-01-01"
    else:
        report["generated_at"] = (now + timedelta(days=1)).isoformat()
    write_json(settings.artifact_dir / "monitoring.json", report)
    with TestClient(create_app(settings, db, clock=lambda: now)) as client:
        text = client.get("/metrics").text
        assert "forecastops_monitoring_fresh 0.0" in text
        assert "forecastops_production_wape NaN" in text
