import argparse
import json
import logging
from pathlib import Path

from filelock import FileLock

from forecastops.config.settings import Settings
from forecastops.data.validate import read_sales
from forecastops.inference.database import Database


def main():
    parser = argparse.ArgumentParser(description="ForecastOps reproducible batch workflows")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ["demo", "generate", "validate", "train", "forecast", "monitor", "cycle"]:
        sub = subs.add_parser(name)
        sub.add_argument("--data", type=Path, default=Path("data/raw/demo.csv"))
        if name in ["demo", "generate"]:
            sub.add_argument("--days", type=int, default=240)
            sub.add_argument("--items", type=int, default=8)
            sub.add_argument("--stores", type=int, default=3)
        if name in ["demo", "train", "cycle"]:
            sub.add_argument(
                "--no-tracking",
                action="store_true",
                help="Local test mode: omit MLflow; retain local audit artifacts",
            )
    m5 = subs.add_parser("ingest-m5")
    m5.add_argument("--directory", type=Path, required=True)
    m5.add_argument("--output", type=Path, default=Path("data/processed/m5.csv"))
    m5.add_argument("--stores", nargs="+", default=["CA_1", "TX_1", "WI_1"])
    m5.add_argument("--items", type=int, default=100)
    m5.add_argument("--days", type=int, default=730)
    args = parser.parse_args()
    settings = Settings()
    settings.prepare()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        with FileLock(str(settings.artifact_dir / "pipeline.lock"), timeout=0):
            result = execute(args, settings)
        print(json.dumps({"event": args.command, **result}, indent=2, default=str))
    except Exception as error:
        logging.error(
            json.dumps(
                {
                    "event": "pipeline_failed",
                    "command": args.command,
                    "error": str(error),
                    "type": type(error).__name__,
                }
            )
        )
        raise SystemExit(1) from error


def execute(args, settings):
    from forecastops.data.synthetic import generate
    from forecastops.inference.batch import forecast
    from forecastops.monitoring.report import monitor
    from forecastops.registry.store import current
    from forecastops.training.pipeline import train

    command = args.command
    if command == "ingest-m5":
        from forecastops.data.m5 import ingest

        data = ingest(args.directory, args.output, args.stores, args.items, args.days)
        return {"rows": len(data), "output": str(args.output)}
    if command in ["generate", "demo"]:
        if args.data.exists():
            raise ValueError("Generation refuses to overwrite an existing dataset; choose --data")
        generate(args.data, args.days, args.stores, args.items, settings.seed)
    data = read_sales(args.data)
    if command in ["generate", "validate"]:
        return {"rows": len(data), "valid": True}
    db = Database(settings.database_url)
    if command in ["train", "demo"]:
        report = train(data, settings, tracking=not args.no_tracking)
        if command == "train":
            return {
                "version": report["version"],
                "candidate": report["name"],
                "quality_gate": report["quality_gate"],
            }
        if not current(settings):
            return {"quality_gate": report["quality_gate"], "status": "no_champion"}
    if command in ["monitor", "cycle"]:
        report = monitor(data, settings, db)
        if command == "monitor":
            return report
        if report["retraining_required"]:
            # Recheck current performance each time; stale request files never cause retraining.
            request_key = f"{report['model_version']}:{report['as_of']}"
            marker = settings.artifact_dir / "last-retrain-request.txt"
            if not marker.exists() or marker.read_text() != request_key:
                train(data, settings, tracking=not args.no_tracking)
                marker.write_text(request_key)
    rows = forecast(data, settings, db)
    if command == "demo":
        monitor(data, settings, db)
    return {
        "forecast_rows": len(rows),
        "model_version": rows[0]["model_version"],
        "forecast_date": rows[0]["forecast_date"],
        "database": "configured forecast store",
    }


if __name__ == "__main__":
    main()
