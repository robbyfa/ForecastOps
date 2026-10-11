# Operational runbook

## Daily operation

1. Append actuals to a validated input panel, retaining the complete history. Keep the data available at a stable, shared path.
2. Run `forecastops cycle --data /path/to/sales.csv` after end-of-day actuals arrive. On first setup, run `train` then `forecast` instead, because monitoring requires a champion.
3. Inspect the JSON process result, `artifacts/monitoring.json`, and service health. A failed command exits nonzero and emits a JSON error event. Native dependency libraries may also write their own logs.
4. Investigate changed series or validation failures before retraining. Inspect store/category errors even if overall performance is acceptable.

Schedule this command through your existing daily scheduler. The repository does not mutate the machine's crontab or configure an AWS schedule. CLI writes are serialized by `artifacts/pipeline.lock`; overlapping jobs fail immediately. Programmatic callers must provide equivalent serialization.

## Promotion and rollback

Reports preserve rejection reasons. Repeated training on the same data cannot replace an incumbent: a new post-training holdout must be available. The registry alias is a discovery convenience, while `champion.json` determines which immutable version the batch job uses. The forecast API serves previously persisted rows and does not load a model for each request.

For rollback, stop workers, retain a copy of the current champion pointer, inspect a previous run's report and artifact, restore that report as `champion.json` atomically, then run `forecastops reconcile` to align the MLflow `champion` alias with its registered version. Issue a new batch and verify `/models/current` and forecast-row versions. Previous batches retain their model versions and issuance times. Save their `batch_id` for exact historical retrieval; selecting a date alone follows that origin's current active pointer.

`champion.json` is the authoritative commit point. Registration and durable run reports precede it. `promotion.json` records pending alias synchronization before the atomic pointer replacement. After a process interruption, recovery reads the committed pointer, never promotes the interrupted journal candidate, and idempotently aligns or removes the discovery alias. CLI training/forecast/monitor/cycle commands retry pending work under the pipeline lock. `forecastops reconcile` also repairs manual alias changes and exits nonzero if the registry is still unavailable.

An alias failure leaves the committed champion usable by immutable URI and emits a warning; it does not roll back promotion. A registry/artifact-server outage can still prevent loading that URI for inference. Inspect `promotion.json` for `pending`, `synced` or `local_only`. Local-only promotion removes a previous tracked alias; an entirely local installation makes no MLflow call. Registry migrations across separate tracking servers require an explicit migration plan; recovery manages the current registry, not historical servers. Keep all programmatic writers serialized as CLI writers are. Do not delete registered models or local artifacts referenced by committed pointers or audit reports.

## Data and storage

Back up Postgres, the MLflow backend/artifacts and the shared artifact directory together. Before upgrading from the legacy `forecasts` schema, stop all old workers and API processes. Do not run old and new writers concurrently.

On initialization, the new application takes a database writer lock (SQLite immediate transaction or PostgreSQL transaction advisory lock), creates `forecast_batches`, `forecast_history`, `forecast_publications` and `forecast_schema`, then copies the legacy table once in the same transaction. Existing row timestamps are preserved and batch headers are marked `legacy_import`; no legacy rows are deleted. A failure rolls back the import and version marker so it can be retried. Invalid legacy batches fail explicitly for review. Data erased by earlier versions cannot be reconstructed.

After migration, verify counts, active model versions and a known historical batch before restarting scheduled work. The old `forecasts` table is retained for inspection but no longer receives writes. Rolling back to an old application binary will not see subsequent publications; restore a coordinated backup or perform a reviewed reverse migration rather than restarting old writers against the new state.

New batches are identified by SHA-256 over canonical dates, series, model version and values. The same content reuses its issuance record; changed content or model version appends a new record. A separate active pointer controls serving. A failed write rolls back both data and pointer changes. These are application-level immutable records, not tamper-proof storage against privileged SQL users. No retention/deletion job is configured.

The local API exposes no administrative writes and is bound to localhost. It currently has no application authentication, TLS termination, quotas, or retention policy. Add them before broader external exposure. Keep the API at one worker until Prometheus multiprocess collection is configured.

For historical inspection, use `forecastops monitor --replay --data ...` and specify `forecast_date` in API requests. Replay never writes a new retraining request or exports live metrics. Live reports require recent actuals and a generation timestamp; `/ready` and default forecast requests require a recent origin and the current model. `/health` remains a service/database check. Monitor the freshness/readiness gauges rather than interpreting missing WAPE as good performance.

Monitoring compares errors and coverage separately by horizon. A single 14-day batch does not provide seven daily observations at each horizon. Existing champions without `holdout_by_horizon` cannot automatically trigger retraining: acquire a properly evaluated new candidate instead of inventing reference metrics or silently rerunning a frozen benchmark. The M5 benchmark artifacts remain unchanged.

PSI references now cover every series' latest 28 training days; prediction PSI uses CV predictions. Seasonality and product mix still affect pooled drift. Monitoring retains matured scores for earlier models in `by_model`; only the current champion can request retraining. It selects the earliest issuance per origin/target/series across versions, never the active replacement batch. Live rows must be issued within the declared origin UTC day; late or future-origin backfills remain available for explicit replay only. This policy does not establish when upstream actuals were available or recover rows erased before migration.

## Isolated service integration check

```bash
docker build -t forecastops:ci .
docker compose -p forecastops-ci -f compose.integration.yml up -d --wait postgres mlflow api
docker compose -p forecastops-ci -f compose.integration.yml run --rm worker
docker compose -p forecastops-ci -f compose.integration.yml run --rm verifier
```

The test project has separate volumes, an ephemeral loopback Postgres port and no published API/MLflow ports. The verifier checks model registration and alias through HTTP, reloads the model for repeated inference, checks Postgres history and timestamps, and retrieves original/replacement batches through the running HTTP API. Synthetic historical dates deliberately return 503 from readiness/default forecast routes and 200 for explicit historical selections. This is not proof of a live ingestion schedule or an AWS deployment.

To run database contracts on its PostgreSQL instance:

```bash
PG_ADDRESS=$(docker compose -p forecastops-ci -f compose.integration.yml port postgres 5432)
FORECASTOPS_TEST_POSTGRES_URL="postgresql+psycopg://forecastops:local-test-only@$PG_ADDRESS/forecastops" \
  uv run pytest tests/integration/test_database.py -q
```

Each PostgreSQL test uses its own temporary schema. CI saves `stack-verification.json` and service logs as artifacts before removing the test stack. For local cleanup after inspection, run `docker compose -p forecastops-ci -f compose.integration.yml down -v --remove-orphans`; this deletes only that disposable test project's volumes. Use a fresh test stack for each complete worker/verifier run.
