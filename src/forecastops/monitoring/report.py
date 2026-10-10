import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import select

from forecastops.data.validate import validate
from forecastops.evaluation.metrics import segmented
from forecastops.inference.database import forecasts
from forecastops.registry.store import current, write_json


def psi(reference, observed):
    reference, observed = np.asarray(reference), np.asarray(observed)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        # Retain a narrow bin around a constant training feature to detect changes.
        center = float(edges[0])
        edges = np.array([-np.inf, center - 1e-6, center + 1e-6, np.inf])
    else:
        edges = np.concatenate(([-np.inf], edges, [np.inf]))
    p = np.maximum(np.histogram(reference, edges)[0] / len(reference), 1e-6)
    q = np.maximum(np.histogram(observed, edges)[0] / len(observed), 1e-6)
    return float(np.sum((q - p) * np.log(q / p)))


def monitor(data, settings, database):
    data = validate(data)
    champion = current(settings)
    if not champion:
        raise ValueError("No champion available for monitoring")
    start = data.date.max() - pd.Timedelta(days=6)
    recent = data[data.date.ge(start)]
    reference = champion["reference"]
    drift = {
        col: {
            "psi": psi(reference[col], recent[col]),
            "ks": float(ks_2samp(reference[col], recent[col]).statistic),
        }
        for col in ["demand", "price"]
    }
    database.initialize()
    with database.engine.connect() as conn:
        statement = select(forecasts).where(
            forecasts.c.model_version == champion["version"],
            forecasts.c.target_date >= start.date(),
            forecasts.c.target_date <= data.date.max().date(),
        )
        stored = pd.DataFrame(conn.execute(statement).mappings().all())
    performance, matched, coverage = None, 0, 0.0
    if not stored.empty:
        stored["target_date"] = pd.to_datetime(stored.target_date)
        # One issued prediction per realized series/day: choose latest available origin.
        stored = stored.sort_values("forecast_date").drop_duplicates(
            ["store_id", "item_id", "target_date"], keep="last"
        )
        scored = stored.merge(
            data.rename(columns={"date": "target_date", "demand": "target"}),
            on=["store_id", "item_id", "target_date"],
            suffixes=("", "_actual"),
            validate="one_to_one",
        )
        if not scored.empty:
            matched = len(scored)
            coverage = matched / len(recent)
            performance = segmented(scored)
    enough = coverage >= 0.8 and matched >= 7
    score = performance["overall"]["wape"] if performance else None
    baseline = champion["holdout"]["candidate"]["overall"]["wape"]
    expected = set(map(tuple, champion["series_keys"]))
    actual = set(map(tuple, data[["store_id", "item_id", "category"]].drop_duplicates().values))
    schema_changed = actual != expected
    degraded = enough and score is not None and score > max(baseline * 1.25, baseline + 0.02)
    report = {
        "model_version": champion["version"],
        "as_of": str(data.date.max().date()),
        "drift": drift,
        "performance": performance,
        "matched_actuals": matched,
        "coverage": coverage,
        "series_changed": schema_changed,
        "status": "ready" if enough else "awaiting_actuals",
        "retraining_required": bool(degraded and not schema_changed),
        "reason": "performance_degraded" if degraded else "no_performance_trigger",
        "drift_alert": any(v["psi"] > 0.2 for v in drift.values()),
    }
    if not stored.empty:
        report["prediction_drift_psi"] = psi(reference["demand"], stored.prediction)
    write_json(settings.artifact_dir / "monitoring.json", report)
    if report["retraining_required"]:
        write_json(settings.artifact_dir / "retrain-request.json", report)
    return report
