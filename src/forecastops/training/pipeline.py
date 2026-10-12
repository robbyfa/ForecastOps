import hashlib
import os
import time
from datetime import UTC, datetime

import joblib
import pandas as pd

from forecastops.data.validate import validate
from forecastops.evaluation.backtest import cutoffs, evaluate, score_frozen
from forecastops.evaluation.intervals import calibrate, coverage
from forecastops.evaluation.metrics import segmented
from forecastops.features.build import FEATURES, build
from forecastops.monitoring.reference import build_reference
from forecastops.registry.gates import quality_gate
from forecastops.registry.store import current, promote, reconcile, write_json
from forecastops.training.models import NAMES, PARAMETERS, fit, predict
from forecastops.training.provenance import provenance


def train(data, settings, tracking=True):
    settings.prepare()
    # CLI/worker must serialize training, inference and monitoring writes.
    reconcile(settings)
    start = time.perf_counter()
    data = validate(data)
    # Selection labels end before calibration; calibration labels end before test.
    holdout = data.date.max() - pd.Timedelta(days=settings.horizon)
    calibration_origins = [
        holdout - pd.Timedelta(days=settings.horizon * i)
        for i in range(settings.calibration_windows, 0, -1)
    ]
    fit_end = calibration_origins[0]
    selection_data = data[data.date.le(fit_end + pd.Timedelta(days=settings.horizon))]
    origins, _ = cutoffs(selection_data, settings.horizon, settings.folds, settings.min_train_days)
    old = current(settings)
    cv, frames = {}, {}
    for name in NAMES:
        frames[name], cv[name] = evaluate(data, name, origins, settings.horizon, settings.seed)
    candidates = [
        n for n in NAMES if n != "seasonal_naive" and cv[n]["overall"]["wape"] is not None
    ]
    winner = min(candidates, key=lambda n: cv[n]["overall"]["wape"]) if candidates else "ridge"
    names = {winner, "seasonal_naive"}
    model = fit(winner, build(data[data.date.le(fit_end)], settings.horizon, True), settings.seed)
    calibration = pd.concat(
        [
            score_frozen(data, winner, model, origin, settings.horizon)[0]
            for origin in calibration_origins
        ],
        ignore_index=True,
    )
    interval = calibrate(calibration, settings.horizon)
    final_results = {
        n: score_frozen(data, n, model if n == winner else None, holdout, settings.horizon)
        for n in names
    }
    interval_metrics, holdout_frame = coverage(final_results[winner][0], interval)
    final_scores = {n: result[1] for n, result in final_results.items()}
    incumbent_score = None
    incumbent_evaluable = old is None or holdout >= pd.Timestamp(
        old.get("data_end", old["train_end"])
    )
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
        gate["reasons"].append("need one complete holdout window after incumbent data cutoff")
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
    bundle = {"name": winner, "model": model, "interval": interval}
    calibration.to_csv(run_dir / "calibration.csv", index=False)
    holdout_frame.to_csv(run_dir / "holdout.csv", index=False)
    with (run_dir / "model.joblib").open("wb") as artifact:
        joblib.dump(bundle, artifact)
        artifact.flush()
        os.fsync(artifact.fileno())
    source = provenance()
    report = {
        "version": version,
        "name": winner,
        "dataset_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        "git_commit": source["git_commit"],
        "provenance": source,
        "incumbent_version": old["version"] if old else None,
        "gate_configuration": {
            "min_improvement": settings.min_improvement,
            "max_segment_regression": settings.max_segment_regression,
            "max_prediction_seconds": settings.max_prediction_seconds,
        },
        "observations": len(data),
        "series": len(data.groupby(["store_id", "item_id"])),
        "train_end": str(fit_end.date()),
        "data_end": str(data.date.max().date()),
        "calibration_origins": [str(d.date()) for d in calibration_origins],
        "calibration_windows": settings.calibration_windows,
        "horizon": settings.horizon,
        "cv_origins": [str(d.date()) for d in origins],
        "holdout_origin": str(holdout.date()),
        "models": cv,
        "parameters": PARAMETERS,
        "holdout": {"candidate": final_scores[winner], "baseline": final_scores["seasonal_naive"]},
        "holdout_by_horizon": {
            str(h): segmented(group) for h, group in final_results[winner][0].groupby("horizon")
        },
        "quality_gate": gate,
        "prediction_seconds": latency,
        "training_seconds": time.perf_counter() - start,
        "features": FEATURES,
        "model_path": str((run_dir / "model.joblib").resolve()),
        "interval": interval,
        "interval_holdout": interval_metrics,
        "interval_method": interval["method"],
        "reference": build_reference(data, frames[winner]),
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
        promote(settings, report)
    return report
