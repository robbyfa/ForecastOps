import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import select

from forecastops.data.validate import validate
from forecastops.evaluation.metrics import segmented
from forecastops.inference.database import forecasts
from forecastops.monitoring.freshness import data_age, utc_now
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


def monitor(data, settings, database, *, now=None, replay=False):
    now = now or utc_now()
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
            forecasts.c.target_date >= start.date(),
            forecasts.c.target_date <= data.date.max().date(),
        )
        stored = pd.DataFrame(conn.execute(statement).mappings().all())
    performance, matched, coverage = None, 0, 0.0
    scored = pd.DataFrame()
    by_model = {}
    if not stored.empty:
        stored["target_date"] = pd.to_datetime(stored.target_date)
        stored["forecast_date"] = pd.to_datetime(stored.forecast_date)
        stored["horizon"] = (stored.target_date - stored.forecast_date).dt.days
        stored = stored[stored.horizon.between(1, 14)]
        if not replay:
            issued = pd.to_datetime(stored.created_at, utc=True)
            # Lead time is measured from origin, not when a late backfill reaches the DB.
            origin_start = stored.forecast_date.dt.tz_localize("UTC")
            stored = stored[
                issued.ge(origin_start)
                & issued.lt(origin_start + pd.Timedelta(days=1))
                & issued.le(now)
            ]
        # Preserve the first timely decision for each origin, even across model changes.
        stored = stored.sort_values(["created_at", "id"]).drop_duplicates(
            ["forecast_date", "target_date", "store_id", "item_id"], keep="first"
        )
        scored = stored.merge(
            data.rename(columns={"date": "target_date", "demand": "target"}),
            on=["store_id", "item_id", "target_date"],
            suffixes=("", "_actual"),
            validate="many_to_one",
        )
        if not scored.empty:
            for version, group in scored.groupby("model_version"):
                by_model[version] = {
                    "matched_actuals": len(group),
                    "performance": segmented(group),
                    "by_horizon": {
                        str(h): {"matched_actuals": len(part), "performance": segmented(part)}
                        for h, part in group.groupby("horizon")
                    },
                }
            scored = scored[
                scored.model_version.eq(champion["version"])
                & scored.horizon.le(champion["horizon"])
            ]
            matched = len(scored)
            performance = segmented(scored) if matched else None
    by_horizon = {}
    references = champion.get("holdout_by_horizon", {})
    degraded_horizons = []
    for horizon in range(1, champion["horizon"] + 1):
        group = scored[scored.horizon.eq(horizon)] if not scored.empty else scored
        count = len(group)
        horizon_coverage = count / len(recent)
        metrics = segmented(group) if count else None
        baseline = references.get(str(horizon), {}).get("overall", {}).get("wape")
        score = metrics["overall"]["wape"] if metrics else None
        enough = horizon_coverage >= 0.8 and count >= 7
        degraded = (
            enough
            and baseline is not None
            and score is not None
            and score > max(baseline * 1.25, baseline + 0.02)
        )
        if degraded:
            degraded_horizons.append(horizon)
        by_horizon[str(horizon)] = {
            "matched_actuals": count,
            "coverage": horizon_coverage,
            "performance": metrics,
            "reference_wape": baseline,
            "status": "awaiting_actuals"
            if not enough
            else ("reference_unavailable" if baseline is None else "ready"),
            "degraded": bool(degraded),
        }
    coverage = matched / (len(recent) * champion["horizon"])
    expected = set(map(tuple, champion["series_keys"]))
    actual = set(map(tuple, data[["store_id", "item_id", "category"]].drop_duplicates().values))
    schema_changed = actual != expected
    age = data_age(str(data.date.max().date()), now)
    fresh = 0 <= age <= settings.max_data_age_days
    eligible = any(v["status"] == "ready" for v in by_horizon.values())
    would_retrain = bool(degraded_horizons and not schema_changed)
    status = (
        "ready"
        if eligible
        else (
            "reference_unavailable"
            if any(v["status"] == "reference_unavailable" for v in by_horizon.values())
            else "awaiting_actuals"
        )
    )
    if not replay and not fresh:
        status = "stale_data"
    report = {
        "model_version": champion["version"],
        "as_of": str(data.date.max().date()),
        "generated_at": now.isoformat(),
        "mode": "replay" if replay else "live",
        "data_age_days": age,
        "drift": drift,
        "performance": performance,
        "aggregate_policy": "pooled available horizon predictions; triggers are horizon-specific",
        "by_horizon": by_horizon,
        "by_model": by_model,
        "issuance_policy": "first prediction issued within the origin UTC day; all models retained",
        "degraded_horizons": degraded_horizons,
        "matched_actuals": matched,
        "coverage": coverage,
        "series_changed": schema_changed,
        "status": status,
        "would_retrain": would_retrain,
        "retraining_required": bool(would_retrain and fresh and not replay),
        "reason": "performance_degraded" if would_retrain else "no_performance_trigger",
        "drift_alert": any(v["psi"] > 0.2 for v in drift.values()),
    }
    if not scored.empty and reference.get("prediction"):
        report["prediction_drift_psi"] = psi(reference["prediction"], scored.prediction)
    write_json(settings.artifact_dir / "monitoring.json", report)
    if report["retraining_required"]:
        write_json(settings.artifact_dir / "retrain-request.json", report)
    return report
