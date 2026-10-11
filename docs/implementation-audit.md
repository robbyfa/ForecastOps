# Implementation audit — 2026-10-11

Reviewed commit: `9c556722cbc6d30a037686585b0b054429f1df9d`.

Verdict: a working local prototype, not yet portfolio-ready as a production forecasting system. The temporal feature/backtest implementation has a sound foundation. Monitoring, provenance and lifecycle recovery need work before real-data headline claims or cloud deployment.

Later on 2026-10-11, the user supplied the M5 files and a separate [first real-data experiment](m5-results.md) was completed. Its product selection was frozen using initial-training price availability and seeded category-stratified sampling, addressing the cohort-selection part of F7 for that experiment. Its measured process peak was approximately 3.96 GiB. This addendum does not mark the production audit findings as fixed; the original observations and their chronology remain below.

This audit changes no production Python code, model pointers, or deployed infrastructure. It adds diagnostic scripts and evidence. Findings below distinguish executed reproductions from inspection-only risks. P1 means fix before relying on the affected production behavior; P2 means address before the corresponding benchmark/deployment claim.

## Remediation update — 2026-10-11

The original findings below are preserved as historical evidence. The subsequent implementation adds:

- **F1:** per-horizon errors, coverage and matching holdout references; a regression with perfect h=1 and failed h=14 now triggers on h=14. Older reports fail closed without horizon references. Scoring still covers only the current champion.
- **F2:** UTC data/report age policies, explicit replay mode, late-backfill exclusion, unavailable stale metrics, `/ready`, historical-date API access and freshness alerts. Input actuals themselves still lack availability/version timestamps.
- **F3:** equal series weighting over the latest 28 training days and genuine CV-prediction drift references. Seasonal/segment-specific drift calibration remains future work.
- **F4:** a durable pending journal, one authoritative atomic champion pointer, alias updates only after commit, automatic pending recovery and explicit `forecastops reconcile`. Fault-injection tests cover failures before the pointer, after commit, after alias update and during rollback recovery; a real SQLite-backed MLflow registry test verifies aliases and their removal.
- **F6:** build revision injection, source and dependency-lock digests, dirty/unknown state, incumbent version and gate settings in training evidence and MLflow.
- **Retraining:** preflight deferral until enough new holdout data exists; failed training still attempts forecasts from the committed champion.
- **F8 (partial):** CI now asserts that the demo actually promoted, aligned its alias, saved the full batch and served it through FastAPI. Live Postgres plus HTTP MLflow startup in CI is still outstanding.

The local suite now has **52 passing tests**. The [updated diagnostic](remediation-evidence.json) shows all 150 reference series represented, stale metrics suppressed, and longer-horizon errors retained. An isolated real MLflow registration/loading smoke run produced and verified 84 forecasts through the database and FastAPI; this used SQLite, not the remote service stack. A Docker build and network-disabled container check verified revision injection, source/lock hashes and promotion-journal writes on Linux. **F5 remains open:** database reruns still replace issued rows at an origin. No new uncertainty calibration, AWS deployment or M5 retraining is claimed. The frozen benchmark and original diagnostic evidence have not been rewritten to reflect newer code. Filesystem recovery assumes shared POSIX storage and serialized writers, not a distributed transaction or arbitrary tracking-server migration.

## Prioritized findings

### F1 — P1: monitoring discards longer-horizon performance

Location: `src/forecastops/monitoring/report.py:53` and `:68`.

For each target day, monitoring keeps only the latest issued prediction. When batches run daily, those are horizon-one predictions. The retraining threshold nevertheless uses pooled 1–14-day holdout WAPE. This compares different forecast tasks and can completely hide degraded replenishment forecasts at longer lead times.

Executed reproduction: daily forecasts with h=1 prediction 10, h=2–14 prediction 100, and actual demand 10. The monitor reports WAPE **0**, coverage **1**, and `retraining_required=false`; h=14 WAPE is **9 (900%)**. See `audit-evidence.json` and `scripts/audit_checks.py`.

Required correction: preserve lead time, calculate metrics and coverage per horizon on matured predictions, and compare to horizon-matched reference errors. Define an explicit business-weighted aggregate if one trigger is needed. Add a regression test where h=1 stays good while h=14 degrades.

### F2 — P1: stale observations continue to look current

Location: `src/forecastops/api/app.py:138` and `src/forecastops/monitoring/report.py:31`.

The metrics endpoint checks model-version equality but never report age. Monitoring anchors its seven-day window to the latest supplied data date, without checking whether ingestion has stopped. A stale report can therefore continue exporting full coverage and a healthy WAPE indefinitely. The API also returns the latest historical forecast batch without a freshness signal; its dates are visible, but `/health` does not indicate data readiness.

Executed reproduction: a report dated **2020-02-29** still exports `forecastops_production_wape 0.0` in the current process; the default API returns a batch dated **2020-02-28**.

Required correction: separate event time, observation availability and report generation time. Export data/report/forecast age and an explicit stale state. Make time injectable for replay tests. Historical backtesting must remain supported explicitly rather than accidentally treating replay data as current production data. Add alerts for missing ingestion and monitoring jobs.

### F3 — P2: the drift reference is a biased tail of sorted series

Location: `src/forecastops/training/pipeline.py:110`.

Validation sorts data by store, item and date. Taking `tail(5000)` then selects the final series, not a representative training distribution. Production monitoring compares every series against this biased reference.

Executed reproduction: a 730-day panel of 3 stores × 50 products has 150 series, but the reference contains only **7 series, all from STORE_3**. Ordinary product/store differences can produce alerts unrelated to drift.

Required correction: use a deterministic stratified reference or, preferably, per-series/segment statistics with explicit weighting and a seasonal comparison window. Persist the sampling policy and selected series. Prediction drift currently compares predictions to historical **actual demand**, not to reference predictions; label that comparison accurately or store a genuine prediction reference.

### F4 — P1: promotion has two independently updated authorities

Location: `src/forecastops/registry/tracking.py:74` and `src/forecastops/training/pipeline.py:125`.

Code inspection: the MLflow champion alias moves before the local report and `champion.json` are published. If a later filesystem write fails, MLflow identifies a new champion while the API/batch pointer remains old or absent. The runbook acknowledges manual reconciliation, but no automatic recovery path or fault-injection test enforces it.

Required correction: define one authoritative promotion record with a pending/committed state and an idempotent recovery procedure. Consumers must use committed versions. Test crashes after model registration, after alias update, and before/after local publication. A filesystem lock prevents overlapping CLI jobs; it does not make remote and local writes atomic.

### F5 — P2: rerunning an origin erases issued-forecast history

Location: `src/forecastops/inference/database.py:58`.

`save()` deletes all rows for an origin and inserts the replacement. This is transactionally safe but cannot preserve both originally issued and replacement predictions. Monitoring claims to evaluate issued forecasts, yet later reruns can change that evidence.

Executed reproduction: publishing `audit-v2` at the same origin as `audit-v1` leaves only v2. The model version is absent from the uniqueness key, and no issuance identifier or historical publication table exists.

Required correction: append immutable batches, retain model version and issuance time, and switch a separate active-publication pointer. Idempotency should identify the same batch, not delete all batches for a date. Score only forecasts available before the relevant decision/target cutoff.

### F6 — P2: container training loses Git provenance

Location: `src/forecastops/training/pipeline.py:83`, `Dockerfile`, and `.github/workflows/deploy.yml:37`.

Code inspection: the image contains source code but neither Git metadata nor a Git executable. The pipeline catches that lookup failure and records `uncommitted`. Deployment tags contain the commit SHA, but the SHA is never passed into the container as build metadata. Consequently, real container training cannot record its source revision using the current mechanism.

Required correction: inject the Git SHA and build/source digest into the image and reports, with a clean/dirty indicator for host runs. Log dependency-lock hash, model configuration and promotion-threshold configuration as well. Avoid claiming commit-only reproducibility for modified working trees.

### F7 — P2: M5 subset and memory policy must be fixed before benchmarking

Location: `src/forecastops/data/m5.py:33`, `:42`, `src/forecastops/features/build.py:42`, and `src/forecastops/training/models.py:33`.

Code inspection: subset selection chooses the first lexicographically sorted eligible items and conditions eligibility on complete prices over the entire requested span, including evaluation dates. This is a deterministic convenience sample with potential category and availability selection bias. It is not evidence of a representative retail benchmark. Freeze the actual item list and eligibility rule before evaluating outcomes, and disclose any retrospective completeness filtering.

Training also expands each origin into fourteen rows, copies feature frames for each horizon and uses dense one-hot encoding. For 300 complete series over 730 days, the final training table has about **2.92 million rows**. With 100 items, 3 stores, 3 categories and 14 numeric features, one float64 design matrix alone is about **2.61 GiB**, excluding frames, transformations and model allocations. This is an analytical estimate, not a measured M5 peak. The 2-GiB Fargate task is unsuitable for that configuration without a measured alternative.

Required correction: start at 50 items on a measured local budget; use a declared category-balanced selection, freeze its manifest and source-file hashes, and profile sparse/categorical encodings and feature construction before scaling. Keep worker resources separate from the API task.

### F8 — P2: CI does not enforce the full operational path

Location: `.github/workflows/ci.yml:29`, `src/forecastops/cli.py:86`, and `tests/integration/test_pipeline_api.py:32`.

Observed: the [initial GitHub CI run](https://github.com/robbyfa/ForecastOps/actions/runs/38066120932) succeeded and its demo produced **84 forecasts**. This is stronger evidence than the earlier README's unobserved-CI statement.

Coverage gap: `demo` returns success with `status=no_champion` if its initial gate rejects the candidate; CI does not assert a published forecast batch. API tests deliberately publish a candidate directly and use SQLite. CI builds Docker and validates Compose but does not start the Postgres/HTTP-MLflow stack or exercise promotion recovery, freshness or daily multi-horizon monitoring.

Required correction: assert the intended positive and negative outcomes separately, then run a small service integration test through HTTP MLflow → batch worker → Postgres → API. Keep production gates intact; do not relax them to force a CI fixture to pass.

## Review of all requested areas

| Area | Assessment |
|---|---|
| Feature leakage | No target-value leakage found in inspected `FEATURES`. Grouped lags/rolls use observations through the end-of-day origin; target calendar is known; future labels are separate. Existing future-perturbation tests pass. Assumes actuals and origin price are available at the stated origin; there is no availability timestamp/versioned data contract. |
| Backtesting | Each fold truncates history before feature creation and fits a fresh preprocessing/model pipeline. Fold prediction windows are adjacent and disjoint; later folds legitimately train on prior validation outcomes. Training labels remain within cutoff. |
| Holdout independence | Within one run, model selection uses CV and the last window is separate. Promotion uses the holdout; the final artifact is then fit on all data, including that window. Its holdout score belongs to the earlier evaluation fit, not a direct test of that final artifact. Repeated tuning is not technically prevented. |
| Champion/challenger | Frozen incumbent comparison avoids testing the incumbent on its own training targets; full post-incumbent holdout is required. Segment checks are MAE, not segment WAPE. Gate settings and compared incumbent version need explicit persistence. Failed gates retain the previous pointer. |
| MLflow lifecycle | Parameters, aggregate metrics, model signature, source package and pinned requirements are logged; batch jobs load immutable version URIs. F4/F6 remain. Per-fold and segment metrics mostly reside in reports/CSV rather than searchable metric series; evaluation predictions should be uploaded as artifacts. |
| Forecast generation | Same feature builder, fixed historical origin, universe checks and finite/nonnegative predictions. Transactional writes and rollback test pass. F5 remains. Latency gate times only an in-memory prediction call, not the end-to-end batch. |
| Monitoring semantics | F1–F3. New champion filtering also omits still-maturing previous-model forecasts, so separate operational scoring by issued model is needed. Global drift is review-only, which avoids automatic retraining on PSI alone. |
| Retraining triggers | Coverage and degradation checks exist; stale request files do not themselves trigger jobs. F1/F2 compromise the signal. A seven-day trigger may run expensive training before the fourteen-day promotion window is available; preflight eligibility should report deferred status. A training exception currently stops that cycle before issuing a fresh champion batch. |
| API behavior | Parameter bounds, stable ordering, pagination, 404s and SQL-error 503 handling are present. Latest batch selection avoids per-item stale fallback, but lacks freshness/readiness semantics. Metrics are single-process and HTTP duration currently stops at response creation rather than completion of body transfer. |
| Docker/CI | Local five-service stack is running; CI is green. F6/F8 remain. Images are not fully digest-pinned. MLflow uses SQLite locally, not an HA backend. API and batch image/resource roles need separation for cloud operation. |
| Terraform | Previously validated configuration, not an applied deployment. External VPC/networking, Postgres, MLflow and deployment identity remain prerequisites. Desired count defaults to zero; ECS stability alone must not be used as proof of a serving API. Add live forecast smoke checks, worker task definition, rollout/rollback tests, TLS/auth ingress and state-backend setup. S3 is provisioned in config but application artifact mirroring is not implemented. No AWS calls or applies were made in this audit. |

## Uncertainty V2 design

Keep model selection, calibration and final evaluation distinct. Choose the model/hyperparameters using early folds, calibrate absolute residuals per horizon using a later untouched calibration block, then evaluate coverage and width on a final untouched test block. A finite-sample conformal-style quantile can be used, but time-series dependence and distribution shift limit exchangeability-based guarantees; report empirical results instead of promising distribution-free temporal coverage.

Persist per-horizon calibration sample count, quantile rule/radius, empirical test coverage, mean interval width, segment coverage and lower-bound clipping policy. Do not choose quantiles or model parameters after looking at the final coverage results. The current winner's pooled CV residuals are not an independent calibration set because those same folds select the winner.

## Next experiments and acceptance criteria

1. Fix F1–F6 with regression/fault-injection tests before calling monitoring and promotion production-ready. Preserve this audit's evidence for before/after comparison.
2. Acquire the official M5 files; user confirmed on 2026-10-11 that they have not been downloaded. No M5 results are fabricated or claimed. Freeze a 3-store, 50-item, approximately 730-day manifest, 14-day horizon and 3–5 fold protocol before training; expand to 100 items after profiling. Record selected IDs/categories, dates, source and canonical dataset hashes, model/gate parameters, fold/horizon/segment/holdout metrics, predictions, elapsed time and peak memory in `docs/m5-results.*`.
3. Add the uncertainty protocol above, preserving a final evaluation block. M5 data are needed for the meaningful coverage headline.
4. Stage a cloud plan only after selecting account/region, budget, networking, Postgres/MLflow hosting, state backend and deployment identity. Run a real data-to-API acceptance check and rollback exercise after deployment. Do not equate provider schema validation with operability.
5. Extend the local measurement below to repeated sustained runs, varied items/stores/page sizes, explicit offered-load levels and sampled worker/container memory. Repeat after cloud deployment.

## Verification and reproduction

Original suite: **26 passed**, with existing dependency deprecation warnings. New diagnostics are separate from the unit suite because they record current undesired behavior rather than assert it as a desired invariant.

```bash
uv run python scripts/audit_checks.py
uv run python scripts/benchmark_api.py --requests 500 --concurrency 8
uv run python scripts/benchmark_pipeline.py
```

The audit and pipeline scripts create temporary datasets/databases/model directories and do not modify the existing champion. The HTTP benchmark issues read-only requests to loopback addresses, with bounded request/concurrency limits.

`docs/local-service-results.json` records a repeated-query measurement through the running Docker API and Postgres: p50 **13.00 ms**, p95 **24.24 ms**, p99 **377.70 ms**, **336.44 requests/s**, and **0/10000 errors**, with concurrency 8 and 20 excluded warmup requests. Total measured duration is **29.72 s**. This is one bounded, warm, closed-loop local run, not a production capacity test or an SLO. The stored result describes the client, endpoint and connection policy.

`docs/local-container-stats.jsonl` records a nearby Docker memory/CPU snapshot, not per-stage peak memory. API: **83.91 MiB**; Postgres: **41.02 MiB**; MLflow: **2.348 GiB**. `docs/local-pipeline-results.json` separately measures host-side synthetic training and SQLite batch wall times plus whole-process peak RSS; MLflow is disabled for that isolated diagnostic, so those durations must not be presented as the production Postgres/remote-registry path.

The isolated host pipeline measured **2.14 s** training wall time, **43.19 ms** batch wall time and **421.25 MiB** whole-process peak RSS for 5,760 synthetic observations and 336 predictions. These figures are not M5 results.
