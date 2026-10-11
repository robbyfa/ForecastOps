# Operational runbook

## Daily operation

1. Append actuals to a validated input panel, retaining the complete history. Keep the data available at a stable, shared path.
2. Run `forecastops cycle --data /path/to/sales.csv` after end-of-day actuals arrive. On first setup, run `train` then `forecast` instead, because monitoring requires a champion.
3. Inspect the JSON process result, `artifacts/monitoring.json`, and service health. A failed command exits nonzero and emits a JSON error event. Native dependency libraries may also write their own logs.
4. Investigate changed series or validation failures before retraining. Inspect store/category errors even if overall performance is acceptable.

Schedule this command through your existing daily scheduler. The repository does not mutate the machine's crontab or configure an AWS schedule. CLI writes are serialized by `artifacts/pipeline.lock`; overlapping jobs fail immediately. Programmatic callers must provide equivalent serialization.

## Promotion and rollback

Reports preserve rejection reasons. Repeated training on the same data cannot replace an incumbent: a new post-training holdout must be available. The registry alias is a discovery convenience, while `champion.json` determines which immutable version the batch job uses. The forecast API serves previously persisted rows and does not load a model for each request.

For rollback, stop workers, retain a copy of the current champion pointer, inspect a previous run's report and artifact, restore that report as `champion.json` atomically, then run `forecastops reconcile` to align the MLflow `champion` alias with its registered version. Issue a new batch and verify `/models/current` and forecast-row versions. Rows retain their model version until replaced at the same origin; immutable issuance history remains an open audit finding.

`champion.json` is the authoritative commit point. Registration and durable run reports precede it. `promotion.json` records pending alias synchronization before the atomic pointer replacement. After a process interruption, recovery reads the committed pointer, never promotes the interrupted journal candidate, and idempotently aligns or removes the discovery alias. CLI training/forecast/monitor/cycle commands retry pending work under the pipeline lock. `forecastops reconcile` also repairs manual alias changes and exits nonzero if the registry is still unavailable.

An alias failure leaves the committed champion usable by immutable URI and emits a warning; it does not roll back promotion. A registry/artifact-server outage can still prevent loading that URI for inference. Inspect `promotion.json` for `pending`, `synced` or `local_only`. Local-only promotion removes a previous tracked alias; an entirely local installation makes no MLflow call. Registry migrations across separate tracking servers require an explicit migration plan; recovery manages the current registry, not historical servers. Keep all programmatic writers serialized as CLI writers are. Do not delete registered models or local artifacts referenced by committed pointers or audit reports.

## Data and storage

Back up Postgres, the MLflow backend/artifacts and the shared artifact directory together. `create_all` bootstraps the initial schema; production schema changes require a migration strategy before rollout. Replacing an origin-date batch is atomic and idempotent but intentionally replaces an earlier model's batch for that origin.

The local API exposes no administrative writes and is bound to localhost. It currently has no application authentication, TLS termination, quotas, or retention policy. Add them before broader external exposure. Keep the API at one worker until Prometheus multiprocess collection is configured.

For historical inspection, use `forecastops monitor --replay --data ...` and specify `forecast_date` in API requests. Replay never writes a new retraining request or exports live metrics. Live reports require recent actuals and a generation timestamp; `/ready` and default forecast requests require a recent origin and the current model. `/health` remains a service/database check. Monitor the freshness/readiness gauges rather than interpreting missing WAPE as good performance.

Monitoring compares errors and coverage separately by horizon. A single 14-day batch does not provide seven daily observations at each horizon. Existing champions without `holdout_by_horizon` cannot automatically trigger retraining: acquire a properly evaluated new candidate instead of inventing reference metrics or silently rerunning a frozen benchmark. The M5 benchmark artifacts remain unchanged.

PSI references now cover every series' latest 28 training days; prediction PSI uses CV predictions. Seasonality and product mix still affect pooled drift. Monitoring currently scores only the current champion; pending earlier-model forecasts and immutable issuance history remain follow-up work. Live scoring uses a conservative UTC cutoff before each target day, but cannot recover originally issued rows already replaced by legacy storage.
