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
from forecastops.registry.store import current


class ForecastRow(BaseModel):
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


def create_app(settings=None, database=None):
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

    @app.get("/models/current")
    def model():
        meta = current(settings)
        if meta is None:
            raise HTTPException(404, "No promoted model")
        return {
            key: meta[key]
            for key in ["version", "name", "train_end", "horizon", "dataset_sha256", "quality_gate"]
        }

    def retrieve(item, store, horizon, forecast_date, limit, offset):
        with retrieval.time():
            rows = db.query(item, store, horizon, forecast_date, limit, offset)
        if not rows:
            raise HTTPException(404, "No forecasts for this selection")
        return rows

    @app.get("/forecasts/items/{item_id}", response_model=list[ForecastRow])
    def item_forecasts(
        item_id: str,
        store_id: str | None = None,
        horizon: int = Query(14, ge=1, le=14),
        forecast_date: date | None = None,
        limit: int = Query(1000, ge=1, le=10000),
        offset: int = Query(0, ge=0),
    ):
        return retrieve(item_id, store_id, horizon, forecast_date, limit, offset)

    @app.get("/forecasts/stores/{store_id}", response_model=list[ForecastRow])
    def store_forecasts(
        store_id: str,
        horizon: int = Query(14, ge=1, le=14),
        forecast_date: date | None = None,
        limit: int = Query(1000, ge=1, le=10000),
        offset: int = Query(0, ge=0),
    ):
        return retrieve(None, store_id, horizon, forecast_date, limit, offset)

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
        if path.exists():
            report = json.loads(path.read_text())
            if meta and report["model_version"] == meta["version"]:
                coverage.set(report["coverage"])
                retrain.set(report["retraining_required"])
                if report["performance"] and report["performance"]["overall"]["wape"] is not None:
                    wape.set(report["performance"]["overall"]["wape"])
                for feature, value in report["drift"].items():
                    drift.labels(feature).set(value["psi"])
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
