"""Profile synthetic or CSV training/SQLite batches in an isolated artifact directory."""

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from forecastops.config.settings import Settings
from forecastops.data.synthetic import generate
from forecastops.data.validate import validate
from forecastops.inference.batch import forecast
from forecastops.inference.database import Database
from forecastops.training.pipeline import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise SystemExit("Refusing to overwrite benchmark evidence; choose a new output path")
    with tempfile.TemporaryDirectory(prefix="forecastops-benchmark-") as directory:
        root = Path(directory)
        data = validate(pd.read_csv(args.data) if args.data else generate(root / "sales.csv"))
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
        result = json.dumps(
            {
                "measured_at": datetime.now(UTC).isoformat(),
                "platform": platform.platform(),
                "dataset": str(args.data)
                if args.data
                else "synthetic: 240 days, 3 stores, 8 items, seed 42",
                "provenance": report["provenance"],
                "cv_wape": {n: m["overall"]["wape"] for n, m in report["models"].items()},
                "holdout": report["holdout"],
                "dataset_sha256": report["dataset_sha256"],
                "git_commit": report["git_commit"],
                "observations": len(data),
                "forecast_rows": len(rows),
                "model": report["name"],
                "horizon": settings.horizon,
                "folds": settings.folds,
                "training_wall_seconds": training_seconds,
                "batch_wall_seconds": batch_seconds,
                "process_peak_rss_mib": maxrss / (1024**2 if sys.platform == "darwin" else 1024),
                "scope": "Host process; MLflow disabled; SQLite batch",
                "training_includes": "CV, holdout, final fit, local artifact/report writes",
                "batch_includes": "Validation, load, features, inference, schema/DB writes",
                "memory_scope": "Whole process peak: imports, training and batch",
            },
            indent=2,
        )

        print(result)
        if args.output:
            args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()
