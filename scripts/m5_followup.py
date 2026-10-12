"""Measure V2 on the already-observed M5 cohort; never overwrite frozen evidence."""

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import pandas as pd

from forecastops.config.settings import Settings
from forecastops.data.validate import validate
from forecastops.evaluation.metrics import segmented
from forecastops.inference.batch import forecast
from forecastops.inference.database import Database
from forecastops.registry.store import write_json
from forecastops.training.pipeline import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/m5-50.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/m5-v2"))
    parser.add_argument("--output", type=Path, default=Path("docs/m5-v2-results.json"))
    args = parser.parse_args()
    if args.artifacts.exists() or args.output.exists():
        raise SystemExit("Refusing to overwrite experiment evidence; choose new output paths")
    data = validate(pd.read_csv(args.data))
    digest = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    original = json.loads(Path("docs/m5-results.json").read_text())
    if digest != original["dataset_sha256"]:
        raise SystemExit("Dataset differs from the frozen M5 cohort")
    settings = Settings(
        _env_file=None,
        artifact_dir=args.artifacts,
        database_url=f"sqlite:///{args.artifacts / 'forecasts.db'}",
        horizon=14,
        folds=3,
        calibration_windows=2,
        min_train_days=70,
        seed=42,
        min_improvement=0.02,
        max_segment_regression=0.1,
        max_prediction_seconds=30,
    )
    start = time.perf_counter()
    report = train(data, settings, tracking=False)
    training_seconds = time.perf_counter() - start
    rows, batch_seconds = [], None
    if report["quality_gate"]["passed"]:
        db = Database(settings.database_url)
        start = time.perf_counter()
        rows = forecast(data, settings, db)
        batch_seconds = time.perf_counter() - start
        pd.DataFrame(rows).to_csv(args.artifacts / "forecasts.csv", index=False)
        db.engine.dispose()
    public_report = {k: v for k, v in report.items() if k != "reference"}
    run_dir = Path(report["model_path"]).parent
    public_report["fold_metrics"] = {}
    for name in report["models"]:
        frame = pd.read_csv(run_dir / f"{name}-cv.csv")
        public_report["fold_metrics"][name] = {
            str(origin): segmented(group) for origin, group in frame.groupby("origin")
        }
    public_report["artifact_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(run_dir.iterdir())
        if path.suffix in {".csv", ".joblib"}
    }
    public_report["reference_summary"] = {
        "policy": report["reference"]["policy"],
        "observations": report["reference"]["observations"],
        "series": len(report["reference"]["sampled_series"]),
        "prediction_count": len(report["reference"]["prediction"]),
    }
    result = {
        "experiment": "V2 engineering comparison on previously observed M5 cohort",
        "independent_test": False,
        "limitations": [
            "Same cohort/test dates already inspected in V1; not new independent ML evidence.",
            "Selection windows shifted earlier to reserve two calibration windows.",
            "Cohort price eligibility used 2016-03-27, after the first two V2 selection origins.",
            "Correlated series and temporal drift preclude a distribution-free coverage guarantee.",
            "MLflow disabled; host process, SQLite batch; single measurement.",
        ],
        "platform": platform.platform(),
        "report": public_report,
        "training_wall_seconds": training_seconds,
        "batch_wall_seconds": batch_seconds,
        "forecast_rows": len(rows),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "model": report["name"],
                "gate": report["quality_gate"],
                "interval_holdout": report["interval_holdout"],
                "peak_rss_mib": result["process_peak_rss_mib"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
