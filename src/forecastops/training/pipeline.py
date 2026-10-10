import hashlib
import subprocess
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from forecastops.data.validate import validate
from forecastops.evaluation.backtest import cutoffs, evaluate
from forecastops.evaluation.metrics import segmented
from forecastops.features.build import FEATURES, build
from forecastops.registry.gates import quality_gate
from forecastops.registry.store import current, write_json
from forecastops.training.models import NAMES, PARAMETERS, fit, predict


def train(data, settings, tracking=True):
    settings.prepare()
    # CLI/worker must serialize training, inference and monitoring writes.
    start = time.perf_counter()
    data = validate(data)
    origins, holdout = cutoffs(data, settings.horizon, settings.folds, settings.min_train_days)
    old = current(settings)
    cv, frames = {}, {}
    for name in NAMES:
        frames[name], cv[name] = evaluate(data, name, origins, settings.horizon, settings.seed)
    candidates = [
        n for n in NAMES if n != "seasonal_naive" and cv[n]["overall"]["wape"] is not None
    ]
    winner = min(candidates, key=lambda n: cv[n]["overall"]["wape"]) if candidates else "ridge"
    names = {winner, "seasonal_naive"}
    final_scores = {
        n: evaluate(data, n, [holdout], settings.horizon, settings.seed)[1] for n in names
    }
    incumbent_score = None
    incumbent_evaluable = old is None or holdout >= pd.Timestamp(old["train_end"])
    if old and incumbent_evaluable:
        frozen = joblib.load(old["model_path"])
        inputs = build(data[data.date.le(holdout)], settings.horizon, False)
        inputs["prediction"] = predict(frozen["name"], frozen["model"], inputs)
        truth = data[["store_id", "item_id", "date", "demand"]].rename(
            columns={"date": "target_date", "demand": "target"}
        )
        incumbent_score = segmented(
            inputs.merge(truth, on=["store_id", "item_id", "target_date"], validate="one_to_one")
        )
    model = fit(winner, build(data, settings.horizon, True), settings.seed)
    future = build(data, settings.horizon, False)
    latency_start = time.perf_counter()
    predict(winner, model, future)
    latency = time.perf_counter() - latency_start
    gate = quality_gate(
        final_scores[winner],
        final_scores["seasonal_naive"],
        incumbent_score,
        settings.min_improvement,
        settings.max_segment_regression,
        latency,
        settings.max_prediction_seconds,
    )
    if not incumbent_evaluable:
        gate["passed"] = False
        gate["reasons"].append("need one complete holdout window after incumbent training cutoff")
    # Require baseline improvement on every CV window, not only on the pooled metric.
    for origin in origins:
        from forecastops.evaluation.metrics import metrics

        a = frames[winner].loc[frames[winner].origin.eq(origin)]
        b = frames["seasonal_naive"].loc[frames["seasonal_naive"].origin.eq(origin)]
        aw, bw = metrics(a.target, a.prediction)["wape"], metrics(b.target, b.prediction)["wape"]
        if aw is None or bw is None or not aw < bw:
            gate["passed"] = False
            gate["reasons"].append(f"does not beat baseline at CV origin {origin.date()}")
    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = settings.artifact_dir / "runs" / version
    run_dir.mkdir(parents=True)
    residuals = np.abs(frames[winner].target - frames[winner].prediction)
    radius = float(np.quantile(residuals, 0.9, method="higher"))
    bundle = {"name": winner, "model": model, "interval_radius": radius}
    joblib.dump(bundle, run_dir / "model.joblib")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
    report = {
        "version": version,
        "name": winner,
        "dataset_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        "git_commit": commit,
        "observations": len(data),
        "series": len(data.groupby(["store_id", "item_id"])),
        "train_end": str(data.date.max().date()),
        "horizon": settings.horizon,
        "cv_origins": [str(d.date()) for d in origins],
        "holdout_origin": str(holdout.date()),
        "models": cv,
        "parameters": PARAMETERS,
        "holdout": {"candidate": final_scores[winner], "baseline": final_scores["seasonal_naive"]},
        "quality_gate": gate,
        "prediction_seconds": latency,
        "training_seconds": time.perf_counter() - start,
        "features": FEATURES,
        "model_path": str((run_dir / "model.joblib").resolve()),
        "interval_radius": radius,
        "interval_method": "pooled CV absolute error 90th percentile",
        "reference": {
            "demand": data.demand.tail(5000).tolist(),
            "price": data.price.tail(5000).tolist(),
        },
        "series_keys": data[["store_id", "item_id", "category"]].drop_duplicates().values.tolist(),
    }
    for name, frame in frames.items():
        frame[
            ["origin", "target_date", "store_id", "item_id", "category", "target", "prediction"]
        ].to_csv(run_dir / f"{name}-cv.csv", index=False)
    if incumbent_score:
        report["holdout"]["incumbent"] = incumbent_score
    if tracking:
        from forecastops.registry.tracking import record

        report.update(record(settings, report, bundle, future, gate["passed"]))
    write_json(run_dir / "report.json", report)
    write_json(settings.artifact_dir / "latest-training.json", report)
    if gate["passed"]:
        write_json(settings.artifact_dir / "champion.json", report)
    return report
