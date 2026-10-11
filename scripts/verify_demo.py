"""Assert a demo actually registered, promoted and served forecasts (never relax gates)."""

import json

import mlflow
from fastapi.testclient import TestClient

from forecastops.api.app import create_app
from forecastops.config.settings import Settings
from forecastops.inference.database import Database
from forecastops.registry.store import current


def main():
    settings = Settings()
    champion = current(settings)
    assert champion and champion["quality_gate"]["passed"], "Demo did not promote a champion"
    latest = json.loads((settings.artifact_dir / "latest-training.json").read_text())
    assert champion["version"] == latest["version"], "Demo did not promote its current run"
    _, name, version = champion["model_uri"].split("/")
    client = mlflow.MlflowClient(tracking_uri=champion["tracking_uri"])
    assert str(client.get_model_version_by_alias(name, "champion").version) == version
    database = Database(settings.database_url)
    rows = database.query(limit=10000)
    assert len(rows) == champion["series"] * champion["horizon"]
    assert {row["model_version"] for row in rows} == {champion["version"]}
    with TestClient(create_app(settings, database)) as api:
        response = api.get(
            f"/forecasts/items/{rows[0]['item_id']}",
            params={
                "store_id": rows[0]["store_id"],
                "forecast_date": str(rows[0]["forecast_date"]),
                "horizon": champion["horizon"],
            },
        )
        assert response.status_code == 200
        assert len(response.json()) == champion["horizon"]
        assert api.get("/models/current").json()["version"] == champion["version"]
    print(
        json.dumps(
            {"verified": True, "forecast_rows": len(rows), "model_uri": champion["model_uri"]}
        )
    )


if __name__ == "__main__":
    main()
