"""Non-destructive diagnostic reproductions; writes only to a temporary directory.

These checks describe current behavior, not desired behavior. Re-run after fixes
and compare the JSON to docs/audit-evidence.json.
"""

import json
import tempfile
from datetime import timedelta
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from forecastops.api.app import create_app
from forecastops.config.settings import Settings
from forecastops.data.synthetic import generate
from forecastops.data.validate import validate
from forecastops.inference.database import Database
from forecastops.monitoring.reference import build_reference
from forecastops.monitoring.report import monitor
from forecastops.registry.store import write_json


def run(root):
    settings = Settings(
        _env_file=None,
        artifact_dir=root / "artifacts",
        database_url=f"sqlite:///{root / 'audit.db'}",
    )
    settings.prepare()
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=60),
            "store_id": "A",
            "item_id": "B",
            "category": "FOODS",
            "demand": 10,
            "price": 1.0,
        }
    )
    champion = {
        "version": "audit-v1",
        "name": "audit",
        "train_end": "2020-01-01",
        "horizon": 14,
        "dataset_sha256": "audit",
        "quality_gate": {},
        "reference": {"demand": [10] * 60, "price": [1.0] * 60},
        "series_keys": [["A", "B", "FOODS"]],
        "holdout": {"candidate": {"overall": {"wape": 0.1}}},
        "holdout_by_horizon": {str(h): {"overall": {"wape": 0.1}} for h in range(1, 15)},
    }
    write_json(settings.artifact_dir / "champion.json", champion)
    db = Database(settings.database_url)
    db.initialize()
    for origin in data.date.iloc[30:-1]:
        rows = []
        for h in range(1, 15):
            rows.append(
                {
                    "forecast_date": origin.date(),
                    "target_date": origin.date() + timedelta(days=h),
                    "store_id": "A",
                    "item_id": "B",
                    "category": "FOODS",
                    "prediction": 10 if h == 1 else 100,
                    "lower_bound": 0,
                    "upper_bound": 110,
                    "model_version": "audit-v1",
                }
            )
        db.save(rows)
    report = monitor(data, settings, db, replay=True)
    with TestClient(create_app(settings, db)) as client:
        metrics = client.get("/metrics").text
        stale_wape = next(
            s for s in metrics.splitlines() if s.startswith("forecastops_production_wape ")
        )
        default_response = client.get("/forecasts/items/B")
    origin = data.date.iloc[-2].date()
    before = db.query(origin=origin)[0]["model_version"]
    replacement = rows.copy()
    replacement = [{**row, "model_version": "audit-v2"} for row in replacement]
    db.save(replacement)
    after = sorted({r["model_version"] for r in db.query(origin=origin)})
    db.engine.dispose()
    panel = validate(generate(root / "panel.csv", days=730, stores=3, items=50))
    reference = build_reference(panel, pd.DataFrame({"prediction": [10.0]}))
    return {
        "monitoring_lead_time": {
            "fixture": "Daily issuance; h=1 prediction 10, h=2..14 prediction 100, actual 10",
            "reported_wape": report["performance"]["overall"]["wape"],
            "horizon_14_wape": 9.0,
            "coverage": report["coverage"],
            "retraining_required": report["retraining_required"],
            "would_retrain_in_replay": report["would_retrain"],
        },
        "freshness": {
            "report_as_of": report["as_of"],
            "exported_metric": stale_wape,
            "default_api_status": default_response.status_code,
            "default_api_body": default_response.json(),
        },
        "batch_replacement": {"previous_version": before, "remaining_versions": after},
        "drift_reference": {
            "total_series": len(panel.groupby(["store_id", "item_id"])),
            "represented_series": len(reference["sampled_series"]),
            "represented_stores": sorted({key[0] for key in reference["sampled_series"]}),
            "rows": reference["observations"],
        },
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="forecastops-audit-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
