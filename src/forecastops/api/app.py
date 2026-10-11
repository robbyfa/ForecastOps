import json
import time
from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from forecastops.config.settings import Settings
from forecastops.inference.database import Database
from forecastops.monitoring.freshness import data_age, report_freshness, utc_now
from forecastops.registry.store import current


class ForecastRow(BaseModel):
    batch_id: str
    forecast_date: date
    target_date: date
    store_id: str
    item_id: str
    category: str
    prediction: float = Field(ge=0)
    lower_bound: float = Field(ge=0)
    upper_bound: float = Field(ge=0)
    model_version: str
    created_at: datetime


def create_app(settings=None, database=None, *, clock=utc_now):
    settings = settings or Settings()
    settings.prepare()
    db = database or Database(settings.database_url)
    registry = CollectorRegistry()
    requests = Counter(
        "forecastops_requests_total", "HTTP requests", ["route", "status"], registry=registry
    )
    duration = Histogram(
        "forecastops_request_seconds", "HTTP duration", ["route"], registry=registry
    )
    retrieval = Histogram("forecastops_retrieval_seconds", "Forecast query time", registry=registry)
    wape = Gauge("forecastops_production_wape", "Observed seven-day WAPE", registry=registry)
    coverage = Gauge("forecastops_actuals_coverage", "Actuals coverage", registry=registry)
    drift = Gauge("forecastops_drift_psi", "Feature PSI", ["feature"], registry=registry)
    champion_gauge = Gauge(
        "forecastops_champion", "Current champion", ["version", "model"], registry=registry
    )
    retrain = Gauge("forecastops_retraining_required", "Retraining trigger", registry=registry)
    fresh_gauge = Gauge(
        "forecastops_monitoring_fresh", "Current live monitoring", registry=registry
    )
    report_age = Gauge("forecastops_report_age_seconds", "Monitoring report age", registry=registry)
    actuals_age = Gauge(
        "forecastops_data_age_days", "Latest actuals age in UTC days", registry=registry
    )
    forecast_age = Gauge("forecastops_forecast_age_days", "Latest origin age", registry=registry)
    forecast_ready = Gauge(
        "forecastops_forecast_ready", "Current champion batch ready", registry=registry
    )
    horizon_wape = Gauge(
        "forecastops_horizon_wape", "Observed WAPE by lead time", ["horizon"], registry=registry
    )
    horizon_coverage = Gauge(
        "forecastops_horizon_coverage",
        "Seven-day panel coverage by lead time",
        ["horizon"],
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(app):
        db.initialize()
        yield
        db.engine.dispose()

    app = FastAPI(title="ForecastOps", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            requests.labels(route, str(status)).inc()
            duration.labels(route).observe(time.perf_counter() - start)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"detail": "Forecast store unavailable"})

    @app.get("/health")
    def health():
        db.healthy()
        return {"status": "ok", "champion_available": current(settings) is not None}

    def batch_ready(rows):
        meta = current(settings)
        age = data_age(str(rows[0]["forecast_date"]), clock()) if rows else None
        return bool(
            rows
            and meta
            and 0 <= age <= settings.max_forecast_age_days
            and rows[0]["model_version"] == meta["version"]
        )

    @app.get("/ready")
    def ready():
        rows = db.query(limit=1)
        if not batch_ready(rows):
            raise HTTPException(503, "No current forecast batch for the committed champion")
        return {"status": "ready", "forecast_date": rows[0]["forecast_date"]}

    @app.get("/models/current")
    def model():
        meta = current(settings)
        if meta is None:
            raise HTTPException(404, "No promoted model")
        return {
            key: meta[key]
            for key in ["version", "name", "train_end", "horizon", "dataset_sha256", "quality_gate"]
        }

    def retrieve(item, store, horizon, forecast_date, limit, offset, batch_id):
        with retrieval.time():
            rows = db.query(item, store, horizon, forecast_date, limit, offset, batch_id)
        if not rows:
            raise HTTPException(404, "No forecasts for this selection")
        if forecast_date is None and batch_id is None and not batch_ready(rows):
            raise HTTPException(
                503, "Forecast batch is stale or superseded; select a date for history"
            )
        return rows

    @app.get("/forecasts/items/{item_id}", response_model=list[ForecastRow])
    def item_forecasts(
        item_id: str,
        store_id: str | None = None,
        horizon: int = Query(14, ge=1, le=14),
        forecast_date: date | None = None,
        limit: int = Query(1000, ge=1, le=10000),
        offset: int = Query(0, ge=0),
        batch_id: str | None = Query(None, pattern="^[a-f0-9]{64}$"),
    ):
        return retrieve(item_id, store_id, horizon, forecast_date, limit, offset, batch_id)

    @app.get("/forecasts/stores/{store_id}", response_model=list[ForecastRow])
    def store_forecasts(
        store_id: str,
        horizon: int = Query(14, ge=1, le=14),
        forecast_date: date | None = None,
        limit: int = Query(1000, ge=1, le=10000),
        offset: int = Query(0, ge=0),
        batch_id: str | None = Query(None, pattern="^[a-f0-9]{64}$"),
    ):
        return retrieve(None, store_id, horizon, forecast_date, limit, offset, batch_id)

    @app.get("/metrics", include_in_schema=False)
    def prometheus():
        champion_gauge.clear()
        meta = current(settings)
        if meta:
            champion_gauge.labels(meta["version"], meta["name"]).set(1)
        path = settings.artifact_dir / "monitoring.json"
        # NaN explicitly denotes unavailable/stale data, rather than a false zero error.
        wape.set(float("nan"))
        coverage.set(0)
        retrain.set(0)
        drift.clear()
        horizon_wape.clear()
        horizon_coverage.clear()
        fresh_gauge.set(0)
        report_age.set(float("nan"))
        actuals_age.set(float("nan"))
        latest = db.query(limit=1)
        forecast_ready.set(batch_ready(latest))
        forecast_age.set(
            data_age(str(latest[0]["forecast_date"]), clock()) if latest else float("nan")
        )
        if path.exists():
            report = json.loads(path.read_text())
            freshness = report_freshness(report, settings, clock())
            for gauge, key in [(report_age, "report_age_seconds"), (actuals_age, "data_age_days")]:
                value = freshness[key]
                gauge.set(value if value is not None else float("nan"))
            if meta and report["model_version"] == meta["version"] and freshness["fresh"]:
                fresh_gauge.set(1)
                coverage.set(report["coverage"])
                retrain.set(report["retraining_required"])
                if report["performance"] and report["performance"]["overall"]["wape"] is not None:
                    wape.set(report["performance"]["overall"]["wape"])
                for feature, value in report["drift"].items():
                    drift.labels(feature).set(value["psi"])
                for horizon, value in report.get("by_horizon", {}).items():
                    horizon_coverage.labels(horizon).set(value["coverage"])
                    score = value["performance"]
                    if score and score["overall"]["wape"] is not None:
                        horizon_wape.labels(horizon).set(score["overall"]["wape"])
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
