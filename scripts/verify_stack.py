"""Exercise HTTP MLflow -> worker -> PostgreSQL -> HTTP API in an isolated stack."""

import json

import mlflow
import requests
from sqlalchemy import func, select

from forecastops.config.settings import Settings
from forecastops.data.validate import read_sales
from forecastops.inference.batch import forecast
from forecastops.inference.database import Database, batches, forecasts
from forecastops.registry.store import current, write_json


def main():
    settings = Settings()
    champion = current(settings)
    assert champion and champion["quality_gate"]["passed"], "Worker did not promote"
    latest = json.loads((settings.artifact_dir / "latest-training.json").read_text())
    assert latest["version"] == champion["version"]
    _, name, version = champion["model_uri"].split("/")
    registry = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
    assert str(registry.get_model_version_by_alias(name, "champion").version) == version
    database = Database(settings.database_url)
    assert database.engine.dialect.name == "postgresql"
    database.initialize()
    rows = database.query(limit=10000)
    expected = champion["series"] * champion["horizon"]
    assert len(rows) == expected
    first_id = rows[0]["batch_id"]
    first_time = rows[0]["created_at"]
    # A second real model download/prediction must reuse the exact published batch.
    rerun = forecast(read_sales("data/raw/demo.csv"), settings, database)
    assert {row["batch_id"] for row in rerun} == {first_id}
    assert database.query()[0]["created_at"] == first_time
    with database.engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(batches)) == 1
    params = {"batch_id": first_id, "store_id": rows[0]["store_id"], "horizon": champion["horizon"]}
    endpoint = f"http://api:8000/forecasts/items/{rows[0]['item_id']}"
    response = requests.get(endpoint, params=params, timeout=20)
    response.raise_for_status()
    assert len(response.json()) == champion["horizon"]
    assert {row["model_version"] for row in response.json()} == {champion["version"]}
    replacement = database.save([{**row, "model_version": "storage-test"} for row in rows])
    assert replacement != first_id
    old = requests.get(endpoint, params=params, timeout=20)
    old.raise_for_status()
    assert old.json() == response.json(), "Replacement changed historical API evidence"
    changed = requests.get(endpoint, params={**params, "batch_id": replacement}, timeout=20)
    changed.raise_for_status()
    assert {row["model_version"] for row in changed.json()} == {"storage-test"}
    database.save(rows)  # Restore active pointer, preserving both batches.
    with database.engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(forecasts)) == 2 * expected
    assert requests.get("http://api:8000/health", timeout=20).status_code == 200
    # This fixture is intentionally historical; a healthy process must not claim live readiness.
    assert requests.get("http://api:8000/ready", timeout=20).status_code == 503
    assert requests.get(endpoint, timeout=20).status_code == 503
    metrics = requests.get("http://api:8000/metrics", timeout=20)
    metrics.raise_for_status()
    assert "forecastops_monitoring_fresh 0.0" in metrics.text
    result = {
        "verified": True,
        "forecast_rows": expected,
        "batch_id": first_id,
        "retained_batches": 2,
        "tracking_transport": "HTTP",
        "database": "PostgreSQL",
        "historical_api": "passed",
        "live_freshness_rejection": "passed",
        "git_commit": champion["git_commit"],
        "provenance": champion["provenance"],
    }
    write_json(settings.artifact_dir / "stack-verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
