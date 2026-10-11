"""Frozen, category-stratified M5 benchmark, isolated from the demo champion.

Run prepare first, then run. Never auto-reselect products based on evaluation results.
"""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from forecastops.config.settings import Settings
from forecastops.data.validate import read_sales, validate
from forecastops.evaluation.backtest import evaluate
from forecastops.evaluation.metrics import metrics, segmented
from forecastops.inference.batch import forecast
from forecastops.inference.database import Database
from forecastops.registry.store import write_json
from forecastops.training.models import PARAMETERS
from forecastops.training.pipeline import train

RAW = Path("data/raw/m5")
DATA = Path("data/processed/m5-50.csv")
PROTOCOL = Path("docs/m5-protocol.json")
STORES = ["CA_1", "TX_1", "WI_1"]
COUNTS = {"FOODS": 17, "HOBBIES": 17, "HOUSEHOLD": 16}
HORIZON, FOLDS, DAYS, SEED = 14, 3, 730, 42


def prepare():
    if PROTOCOL.exists() or DATA.exists():
        raise FileExistsError("Protocol or data already exists; do not silently replace the cohort")
    calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    cols = pd.read_csv(RAW / "sales_train_evaluation.csv", nrows=0).columns
    days = [c for c in cols if c.startswith("d_")][-DAYS:]
    dates = calendar[calendar.d.isin(days)]
    cutoff = dates.date.max() - pd.Timedelta(days=HORIZON * (FOLDS + 1))
    weeks = dates.loc[dates.date.le(cutoff), "wm_yr_wk"].unique()
    parts = []
    for chunk in pd.read_csv(
        RAW / "sales_train_evaluation.csv",
        chunksize=1000,
        usecols=["item_id", "store_id", "cat_id", *days],
    ):
        parts.append(chunk[chunk.store_id.isin(STORES)])
    wide = pd.concat(parts, ignore_index=True)
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices.store_id.isin(STORES)]
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("Duplicate prices")
    initial = prices[
        prices.wm_yr_wk.isin(weeks) & prices.sell_price.gt(0) & np.isfinite(prices.sell_price)
    ]
    complete = initial.groupby(["item_id", "store_id"]).wm_yr_wk.nunique().eq(len(weeks))
    eligible = complete.groupby(level="item_id").sum()
    eligible = set(eligible[eligible.eq(len(STORES))].index)
    catalog = wide[["item_id", "cat_id"]].drop_duplicates()
    rng = np.random.default_rng(SEED)
    selected, eligible_counts = [], {}
    for category, count in COUNTS.items():
        candidates = sorted(
            catalog.loc[catalog.cat_id.eq(category) & catalog.item_id.isin(eligible), "item_id"]
        )
        eligible_counts[category] = len(candidates)
        if len(candidates) < count:
            raise ValueError(f"Insufficient training-eligible products in {category}")
        selected.extend(sorted(rng.choice(candidates, count, replace=False).tolist()))
    protocol = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "source": "sales_train_evaluation.csv",
        "source_files": json.loads(Path("docs/m5-inputs.json").read_text())["files"],
        "stores": STORES,
        "category_counts": COUNTS,
        "items": selected,
        "eligible_counts": eligible_counts,
        "seed": SEED,
        "days": DAYS,
        "start_date": str(dates.date.min().date()),
        "end_date": str(dates.date.max().date()),
        "horizon": HORIZON,
        "folds": FOLDS,
        "selection_cutoff": str(cutoff.date()),
        "parameters": PARAMETERS,
        "eligibility": "Positive finite prices for every initial-training week in all stores",
        "selection": "Seeded category sampling from sorted IDs; no demand filtering",
        "missing_evaluation_prices": "Fail; no replacement products and no imputation",
        "holdout_policy": "Final 14 days for one promotion evaluation; no tuning on results",
        "availability_assumption": "Sales and weekly price known at end-of-day origin",
        "limitations": "Balanced mature-product cohort; not full M5 hierarchy or official WRMSSE",
    }
    # Freeze selection before joining evaluation-period prices or evaluating any predictions.
    write_json(PROTOCOL, protocol)
    panel = wide[wide.item_id.isin(selected)].melt(
        id_vars=["store_id", "item_id", "cat_id"], var_name="d", value_name="demand"
    )
    panel = panel.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", validate="many_to_one")
    panel = panel.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one"
    )
    panel = validate(panel.rename(columns={"cat_id": "category", "sell_price": "price"}))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(DATA, index=False)
    protocol["canonical_dataset_sha256"] = hashlib.sha256(
        panel.to_csv(index=False).encode()
    ).hexdigest()
    protocol["observations"] = len(panel)
    protocol["series"] = len(panel.groupby(["store_id", "item_id"]))
    write_json(PROTOCOL, protocol)
    print(
        json.dumps(
            {
                k: protocol[k]
                for k in [
                    "observations",
                    "series",
                    "start_date",
                    "end_date",
                    "selection_cutoff",
                    "eligible_counts",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


def run():
    protocol = json.loads(PROTOCOL.read_text())
    data = read_sales(DATA)
    digest = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    if digest != protocol["canonical_dataset_sha256"]:
        raise ValueError("Input no longer matches frozen protocol")
    if protocol["parameters"] != PARAMETERS:
        raise ValueError("Model parameters changed after protocol freeze")
    output = Path("artifacts/m5")
    if (output / "latest-training.json").exists():
        raise FileExistsError("This protocol has already run; retain the first evaluation")
    settings = Settings(
        _env_file=None,
        artifact_dir=output,
        database_url="sqlite:///artifacts/m5/forecasts.db",
        tracking_uri="sqlite:///artifacts/m5/mlflow.db",
        horizon=HORIZON,
        folds=FOLDS,
        seed=SEED,
        min_train_days=70,
        min_improvement=0.02,
        max_segment_regression=0.1,
        max_prediction_seconds=30,
    )
    start = time.perf_counter()
    report = train(data, settings, tracking=True)
    training_wall = time.perf_counter() - start
    run_dir = output / "runs" / report["version"]
    fold_metrics = {}
    for name in report["models"]:
        frame = pd.read_csv(run_dir / f"{name}-cv.csv")
        fold_metrics[name] = {
            str(origin): segmented(group) for origin, group in frame.groupby("origin")
        }
    holdout_artifacts, horizon_metrics = {}, {}
    for name in {report["name"], "seasonal_naive"}:
        frame, _ = evaluate(data, name, [pd.Timestamp(report["holdout_origin"])], HORIZON, SEED)
        path = run_dir / f"{name}-holdout.csv"
        frame[
            [
                "origin",
                "target_date",
                "horizon",
                "store_id",
                "item_id",
                "category",
                "target",
                "prediction",
            ]
        ].to_csv(path, index=False)
        holdout_artifacts[name] = str(path)
        horizon_metrics[name] = {
            str(h): metrics(group.target, group.prediction) for h, group in frame.groupby("horizon")
        }
    batch_rows, batch_seconds = 0, None
    if report["quality_gate"]["passed"]:
        db = Database(settings.database_url)
        start = time.perf_counter()
        rows = forecast(data, settings, db)
        batch_seconds = time.perf_counter() - start
        db.engine.dispose()
        pd.DataFrame(rows).to_csv(run_dir / "forecasts.csv", index=False)
        batch_rows = len(rows)
    source_digest = hashlib.sha256()
    for path in sorted(
        [*Path("src").rglob("*.py"), Path(__file__), Path("pyproject.toml"), Path("uv.lock")]
    ):
        source_digest.update(str(path).encode())
        source_digest.update(path.read_bytes())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "measured_at": datetime.now(UTC).isoformat(),
        "protocol": str(PROTOCOL),
        "dataset_type": "M5 supplied competition files",
        "dataset_sha256": digest,
        "git_commit": report["git_commit"],
        "source_digest": source_digest.hexdigest(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
        "platform": platform.platform(),
        "observations": len(data),
        "series": protocol["series"],
        "name": report["name"],
        "version": report["version"],
        "parameters": PARAMETERS,
        "cv_origins": report["cv_origins"],
        "holdout_origin": report["holdout_origin"],
        "cv_metrics": report["models"],
        "fold_metrics": fold_metrics,
        "holdout_metrics": report["holdout"],
        "holdout_horizon_metrics": horizon_metrics,
        "quality_gate": report["quality_gate"],
        "model_uri": report["model_uri"],
        "gate_settings": {
            "min_improvement": settings.min_improvement,
            "max_segment_regression": settings.max_segment_regression,
            "max_prediction_seconds": settings.max_prediction_seconds,
        },
        "training_wall_seconds_including_tracking": training_wall,
        "batch_wall_seconds": batch_seconds,
        "forecast_rows": batch_rows,
        "process_peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
        "prediction_seconds": report["prediction_seconds"],
        "artifact_directory": str(run_dir),
        "holdout_artifacts": holdout_artifacts,
        "interval_method": report["interval_method"],
        "limitations": [
            "Single frozen experiment; not official M5 competition scoring",
            "Audit issues remain; do not claim production readiness",
            "Future batch emitted only if all existing quality gates pass",
            "Memory peak covers training, tracking and artifact reporting",
        ],
    }
    write_json(Path("docs/m5-results.json"), result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "name",
                    "quality_gate",
                    "forecast_rows",
                    "training_wall_seconds_including_tracking",
                    "process_peak_rss_mib",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    args = parser.parse_args()
    prepare() if args.command == "prepare" else run()
