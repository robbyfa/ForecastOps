# Operational runbook

## Daily operation

1. Append actuals to a validated input panel, retaining the complete history. Keep the data available at a stable, shared path.
2. Run `forecastops cycle --data /path/to/sales.csv` after end-of-day actuals arrive. On first setup, run `train` then `forecast` instead, because monitoring requires a champion.
3. Inspect the JSON process result, `artifacts/monitoring.json`, and service health. A failed command exits nonzero and emits a JSON error event. Native dependency libraries may also write their own logs.
4. Investigate changed series or validation failures before retraining. Inspect store/category errors even if overall performance is acceptable.

Schedule this command through your existing daily scheduler. The repository does not mutate the machine's crontab or configure an AWS schedule. CLI writes are serialized by `artifacts/pipeline.lock`; overlapping jobs fail immediately. Programmatic callers must provide equivalent serialization.

## Promotion and rollback

Reports preserve rejection reasons. Repeated training on the same data cannot replace an incumbent: a new post-training holdout must be available. The registry alias is a discovery convenience, while `champion.json` determines which immutable version the batch job uses. The forecast API serves previously persisted rows and does not load a model for each request.

For rollback, stop workers, retain a copy of the current champion pointer, inspect a previous run's report and artifact, restore that report as `champion.json` atomically, and align the MLflow `champion` alias with its registered version. Issue a new batch and verify `/models/current` and forecast-row versions. Historical rows retain their original model version.

MLflow updates and filesystem pointer changes are not one distributed transaction. If a crash occurs during promotion, compare the MLflow alias with `champion.json`. The pointer remains the batch job's source of truth. Reconcile them deliberately before restarting workers. Do not manually delete a registered model or local artifact still referenced by a champion or historical audit report.

## Data and storage

Back up Postgres, the MLflow backend/artifacts and the shared artifact directory together. `create_all` bootstraps the initial schema; production schema changes require a migration strategy before rollout. Replacing an origin-date batch is atomic and idempotent but intentionally replaces an earlier model's batch for that origin.

The local API exposes no administrative writes and is bound to localhost. It currently has no application authentication, TLS termination, quotas, or retention policy. Add them before broader external exposure. Keep the API at one worker until Prometheus multiprocess collection is configured.

Monitoring's PSI thresholds are heuristics. Its pooled reference is a bounded sample of the training panel; its global feature comparisons can be affected by changes in the product mix. The seven-day rolling score mixes lead times if several batches exist. Use lead-time-specific metrics and seasonal references before treating alerts as production paging criteria.
