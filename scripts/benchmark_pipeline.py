"""Measure an isolated synthetic training/SQLite batch job without changing the champion."""

import json
import platform
import resource
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from forecastops.config.settings import Settings
from forecastops.data.synthetic import generate
from forecastops.data.validate import validate
from forecastops.inference.batch import forecast
from forecastops.inference.database import Database
from forecastops.training.pipeline import train


def main():
    with tempfile.TemporaryDirectory(prefix="forecastops-benchmark-") as directory:
        root = Path(directory)
        data = validate(generate(root / "sales.csv"))
        settings = Settings(
            _env_file=None,
            artifact_dir=root / "artifacts",
            database_url=f"sqlite:///{root / 'forecasts.db'}",
        )
        start = time.perf_counter()
        report = train(data, settings, tracking=False)
        training_seconds = time.perf_counter() - start
        if not report["quality_gate"]["passed"]:
            raise SystemExit("Gate rejected candidate; batch measurement not performed")
        db = Database(settings.database_url)
        start = time.perf_counter()
        rows = forecast(data, settings, db)
        batch_seconds = time.perf_counter() - start
        db.engine.dispose()
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            json.dumps(
                {
                    "measured_at": datetime.now(UTC).isoformat(),
                    "platform": platform.platform(),
                    "dataset": "synthetic: 240 days, 3 stores, 8 items, seed 42",
                    "dataset_sha256": report["dataset_sha256"],
                    "git_commit": report["git_commit"],
                    "observations": len(data),
                    "forecast_rows": len(rows),
                    "model": report["name"],
                    "horizon": settings.horizon,
                    "folds": settings.folds,
                    "training_wall_seconds": training_seconds,
                    "batch_wall_seconds": batch_seconds,
                    "process_peak_rss_mib": maxrss
                    / (1024**2 if sys.platform == "darwin" else 1024),
                    "scope": "Host process; MLflow disabled; SQLite batch",
                    "training_includes": "CV, holdout, final fit, local artifact/report writes",
                    "batch_includes": "Validation, load, features, inference, schema/DB writes",
                    "memory_scope": "Whole process peak: imports, training and batch",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
